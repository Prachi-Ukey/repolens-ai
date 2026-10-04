import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.models.repository import Repository
from app.models.file import RepositoryFile
from app.models.chunk import CodeChunk
from app.models.job import AnalysisJob
from app.services.auth_service import get_current_user
from app.github.parser import parse_file_into_chunks, compute_sha256
from app.ai.vector_store import get_vector_store

router = APIRouter(prefix="/demo", tags=["Demo Dataset"])

SAMPLE_REPO_DATA = {
    "url": "https://github.com/repolens/demo-chat-app",
    "owner": "repolens",
    "name": "demo-chat-app",
    "description": "Pre-packaged full-stack real-time chat application demo dataset with JWT authentication, WebSockets, and MongoDB.",
    "primary_language": "JavaScript",
    "files": [
        {
            "path": "server/index.js",
            "name": "index.js",
            "language": "javascript",
            "content": """const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const authRoutes = require('./routes/auth');
const connectDB = require('./config/db');

const app = express();
app.use(express.json());

connectDB();

app.use('/api/auth', authRoutes);

const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });

io.on('connection', (socket) => {
  console.log('User connected:', socket.id);
  socket.on('send_message', (data) => {
    io.emit('receive_message', data);
  });
});

server.listen(5000, () => console.log('Server running on port 5000'));
"""
        },
        {
            "path": "server/middleware/authMiddleware.js",
            "name": "authMiddleware.js",
            "language": "javascript",
            "content": """const jwt = require('jsonwebtoken');

module.exports = function verifyToken(req, res, next) {
  const token = req.header('Authorization')?.replace('Bearer ', '');
  if (!token) {
    return res.status(401).json({ message: 'Access denied. No token provided.' });
  }

  try {
    const verified = jwt.verify(token, process.env.JWT_SECRET || 'secretkey');
    req.user = verified;
    next();
  } catch (err) {
    res.status(400).json({ message: 'Invalid or expired token.' });
  }
};
"""
        },
        {
            "path": "server/controllers/authController.js",
            "name": "authController.js",
            "language": "javascript",
            "content": """const User = require('../models/User');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

exports.register = async (req, res) => {
  const { username, email, password } = req.body;
  const salt = await bcrypt.genSalt(10);
  const hashedPassword = await bcrypt.hash(password, salt);

  const newUser = new User({ username, email, password: hashedPassword });
  await newUser.save();
  res.status(201).json({ message: 'User registered successfully' });
};

exports.login = async (req, res) => {
  const { email, password } = req.body;
  const user = await User.findOne({ email });
  if (!user) return res.status(400).json({ message: 'User not found' });

  const validPassword = await bcrypt.compare(password, user.password);
  if (!validPassword) return res.status(400).json({ message: 'Invalid password' });

  const token = jwt.sign({ id: user._id, username: user.username }, process.env.JWT_SECRET, { expiresIn: '1d' });
  res.json({ token, user: { id: user._id, username: user.username, email: user.email } });
};
"""
        },
        {
            "path": "server/config/db.js",
            "name": "db.js",
            "language": "javascript",
            "content": """const mongoose = require('mongoose');

const connectDB = async () => {
  try {
    const conn = await mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/demochat');
    console.log(`MongoDB Connected: ${conn.connection.host}`);
  } catch (err) {
    console.error(`Error: ${err.message}`);
    process.exit(1);
  }
};

module.exports = connectDB;
"""
        },
        {
            "path": "client/src/App.jsx",
            "name": "App.jsx",
            "language": "javascript",
            "content": """import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';

const socket = io('http://localhost:5000');

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  useEffect(() => {
    socket.on('receive_message', (msg) => {
      setMessages((prev) => [...prev, msg]);
    });
  }, []);

  const sendMessage = () => {
    if (input.trim()) {
      socket.emit('send_message', { text: input, timestamp: new Date() });
      setInput('');
    }
  };

  return (
    <div className="chat-container">
      <h2>RepoLens Demo Chat App</h2>
      <div className="message-box">
        {messages.map((m, i) => <div key={i}>{m.text}</div>)}
      </div>
      <input value={input} onChange={(e) => setInput(e.target.value)} />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}
"""
        }
    ]
}

@router.post("/load-sample")
def load_sample_repository(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    existing = db.query(Repository).filter(
        Repository.user_id == current_user.id,
        Repository.name == SAMPLE_REPO_DATA["name"]
    ).first()

    if existing:
        return {"message": "Demo repository already loaded.", "repository_id": existing.id}

    repo = Repository(
        user_id=current_user.id,
        url=SAMPLE_REPO_DATA["url"],
        owner=SAMPLE_REPO_DATA["owner"],
        name=SAMPLE_REPO_DATA["name"],
        default_branch="main",
        primary_language=SAMPLE_REPO_DATA["primary_language"],
        description=SAMPLE_REPO_DATA["description"],
        status="completed",
        file_count=len(SAMPLE_REPO_DATA["files"])
    )
    db.add(repo)
    db.flush()

    all_chunks = []
    file_objs = []
    for f in SAMPLE_REPO_DATA["files"]:
        content = f["content"]
        line_count = len(content.splitlines())
        sha = compute_sha256(content)

        file_obj = RepositoryFile(
            repository_id=repo.id,
            file_path=f["path"],
            file_name=f["name"],
            language=f["language"],
            size_bytes=len(content.encode("utf-8")),
            line_count=line_count,
            sha_hash=sha,
            content=content
        )
        db.add(file_obj)
        db.flush()

        chunks = parse_file_into_chunks(f["path"], content, f["language"])
        for c in chunks:
            chunk_obj = CodeChunk(
                repository_id=repo.id,
                file_id=file_obj.id,
                file_path=c["file_path"],
                language=c["language"],
                start_line=c["start_line"],
                end_line=c["end_line"],
                symbol_name=c["symbol_name"],
                chunk_id=c["chunk_id"],
                content=c["content"],
                sha_hash=c["sha_hash"]
            )
            all_chunks.append(c)
            db.add(chunk_obj)

    # Store embeddings in Chroma
    vector_store = get_vector_store()
    vector_store.upsert_chunks(repo.id, all_chunks)

    summary = {
        "overview": f"Demo Repository '{repo.name}' pre-loaded with {len(SAMPLE_REPO_DATA['files'])} core files.",
        "primary_language": repo.primary_language,
        "total_files": len(SAMPLE_REPO_DATA["files"]),
        "total_chunks": len(all_chunks),
        "is_demo": True
    }
    repo.summary_json = json.dumps(summary)
    repo.chunk_count = len(all_chunks)

    # Job
    job = AnalysisJob(
        repository_id=repo.id,
        status="completed",
        current_stage="completed",
        progress_percent=100,
        message="Demo dataset ready."
    )
    db.add(job)
    db.commit()

    return {"message": "Demo repository loaded successfully.", "repository_id": repo.id}
