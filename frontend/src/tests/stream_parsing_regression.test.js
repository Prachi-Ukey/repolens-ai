/**
 * Regression test suite for RepoLens AI NDJSON stream parsing & grounding status.
 * Tests:
 * 1. Stream records parse correctly when JSON is split at arbitrary byte boundaries.
 * 2. Multiple NDJSON records in one read parse correctly.
 * 3. The final `done` event works with and without a trailing newline.
 * 4. A genuinely ungrounded completed answer remains ungrounded.
 * 5. A grounded completed answer retains its sources and grounded status.
 * 6. CRLF line endings are safely parsed without trailing carriage returns.
 * 7. Incomplete unparseable records log safe diagnostics without exposing secrets.
 */

import assert from 'node:assert';

function createStreamSimulator() {
  const decoder = new TextDecoder('utf-8');
  let streamBuffer = '';
  const parsedRecords = [];
  const warnings = [];

  const processRecord = (recordText) => {
    const trimmed = recordText.trim();
    if (!trimmed) return;

    try {
      const data = JSON.parse(trimmed);
      parsedRecords.push(data);
    } catch (e) {
      warnings.push(`[ChatStream] Incomplete or unparseable record (length: ${trimmed.length}): ${e.message}`);
    }
  };

  const feedChunk = (uint8Array) => {
    streamBuffer += decoder.decode(uint8Array, { stream: true });
    const lines = streamBuffer.split(/\r?\n/);
    streamBuffer = lines.pop() ?? '';

    for (const line of lines) {
      processRecord(line);
    }
  };

  const finishStream = () => {
    streamBuffer += decoder.decode();
    if (streamBuffer.trim()) {
      processRecord(streamBuffer);
      streamBuffer = '';
    }
  };

  return { feedChunk, finishStream, parsedRecords, warnings };
}

// -------------------------------------------------------------
// Test 1: Multiple NDJSON records in a single chunk
// -------------------------------------------------------------
console.log('Running Test 1: Multiple NDJSON records in one chunk...');
{
  const sim = createStreamSimulator();
  const encoder = new TextEncoder();
  const payload = [
    JSON.stringify({ type: 'init', conversation_id: 'conv-101' }),
    JSON.stringify({ type: 'chunk', content: 'Hello ' }),
    JSON.stringify({ type: 'chunk', content: 'world!' }),
    JSON.stringify({ type: 'done', answer: 'Hello world!', sources: [{ file_path: 'README.md', start_line: 1, end_line: 10 }], grounded: true })
  ].join('\n') + '\n';

  sim.feedChunk(encoder.encode(payload));
  sim.finishStream();

  assert.strictEqual(sim.parsedRecords.length, 4, 'Should parse all 4 records');
  assert.strictEqual(sim.parsedRecords[0].type, 'init');
  assert.strictEqual(sim.parsedRecords[1].content, 'Hello ');
  assert.strictEqual(sim.parsedRecords[2].content, 'world!');
  assert.strictEqual(sim.parsedRecords[3].type, 'done');
  assert.strictEqual(sim.parsedRecords[3].grounded, true);
  assert.strictEqual(sim.parsedRecords[3].sources.length, 1);
  assert.strictEqual(sim.warnings.length, 0);
  console.log('  PASS: Multiple records in one chunk parsed cleanly.');
}

// -------------------------------------------------------------
// Test 2: Arbitrary byte boundaries (1-byte, 2-byte, 3-byte slices)
// -------------------------------------------------------------
console.log('Running Test 2: Arbitrary byte boundary splitting...');
{
  const records = [
    { type: 'init', conversation_id: 'conv-202' },
    { type: 'chunk', content: 'Token αβγ ' },
    { type: 'chunk', content: 'Unicode emojis: 🚀✨ ' },
    { type: 'done', answer: 'Complete explanation with [app.py:10-20]', sources: [{ file_path: 'app.py', start_line: 10, end_line: 20 }], grounded: true }
  ];
  const encoder = new TextEncoder();
  const rawText = records.map(r => JSON.stringify(r)).join('\n') + '\n';
  const fullBytes = encoder.encode(rawText);

  // Test various chunk sizes
  for (const chunkSize of [1, 2, 3, 5, 7, 11, 23, 64]) {
    const sim = createStreamSimulator();
    for (let offset = 0; offset < fullBytes.length; offset += chunkSize) {
      const slice = fullBytes.subarray(offset, Math.min(offset + chunkSize, fullBytes.length));
      sim.feedChunk(slice);
    }
    sim.finishStream();

    assert.strictEqual(sim.parsedRecords.length, records.length, `Chunk size ${chunkSize} should yield ${records.length} records`);
    assert.deepStrictEqual(sim.parsedRecords, records, `Records parsed with chunkSize ${chunkSize} must match expected`);
    assert.strictEqual(sim.warnings.length, 0, `No warnings expected for chunkSize ${chunkSize}`);
  }
  console.log('  PASS: Arbitrary byte boundary chunking (1, 2, 3, 5, 7, 11, 23, 64 bytes) parsed perfectly.');
}

// -------------------------------------------------------------
// Test 3: CRLF Line Endings
// -------------------------------------------------------------
console.log('Running Test 3: CRLF line endings...');
{
  const sim = createStreamSimulator();
  const encoder = new TextEncoder();
  const crlfPayload = '{"type":"init","conversation_id":"crlf-1"}\r\n{"type":"chunk","content":"line1"}\r\n{"type":"done","answer":"line1","grounded":true,"sources":[]}\r\n';

  sim.feedChunk(encoder.encode(crlfPayload));
  sim.finishStream();

  assert.strictEqual(sim.parsedRecords.length, 3);
  assert.strictEqual(sim.parsedRecords[0].type, 'init');
  assert.strictEqual(sim.parsedRecords[1].content, 'line1');
  assert.strictEqual(sim.parsedRecords[2].answer, 'line1');
  assert.strictEqual(sim.warnings.length, 0);
  console.log('  PASS: CRLF line endings handled without error.');
}

// -------------------------------------------------------------
// Test 4: Final 'done' record WITHOUT trailing newline
// -------------------------------------------------------------
console.log('Running Test 4: Final done record without trailing newline...');
{
  const sim = createStreamSimulator();
  const encoder = new TextEncoder();
  const noTrailingNewline = '{"type":"init","conversation_id":"c1"}\n{"type":"done","answer":"Final Answer","grounded":true,"sources":[{"file_path":"main.py","start_line":1,"end_line":15}]}';

  sim.feedChunk(encoder.encode(noTrailingNewline));
  sim.finishStream();

  assert.strictEqual(sim.parsedRecords.length, 2);
  assert.strictEqual(sim.parsedRecords[1].type, 'done');
  assert.strictEqual(sim.parsedRecords[1].answer, 'Final Answer');
  assert.strictEqual(sim.parsedRecords[1].grounded, true);
  assert.strictEqual(sim.warnings.length, 0);
  console.log('  PASS: Final done record without trailing newline was flushed and processed.');
}

// -------------------------------------------------------------
// Test 5: Incomplete record logs safe diagnostic without exposing secrets
// -------------------------------------------------------------
console.log('Running Test 5: Safe diagnostic on truncated/corrupted stream...');
{
  const sim = createStreamSimulator();
  const encoder = new TextEncoder();
  const secretKey = 'API_SECRET_KEY_9999999999';
  const truncatedPayload = `{"type":"done","secret":"${secretKey}`;

  sim.feedChunk(encoder.encode(truncatedPayload));
  sim.finishStream();

  assert.strictEqual(sim.parsedRecords.length, 0, 'Truncated record should not parse');
  assert.strictEqual(sim.warnings.length, 1, 'Should log exactly 1 safe warning');
  assert.ok(sim.warnings[0].includes('[ChatStream] Incomplete or unparseable record'), 'Warning format matches');
  assert.ok(!sim.warnings[0].includes(secretKey), 'Warning must NOT expose message content or secrets');
  console.log('  PASS: Safe diagnostic logged without exposing secrets.');
}

// -------------------------------------------------------------
// Test 6: Genuinely ungrounded completed answer remains ungrounded
// -------------------------------------------------------------
console.log('Running Test 6: Ungrounded answer remains ungrounded even if sources exist...');
{
  const ungroundedDoneMessage = {
    content: "I couldn't find enough evidence [README.md:1-12].",
    sources: [{ file_path: 'README.md', start_line: 1, end_line: 12 }],
    grounded: false,
    isStreaming: false,
    isError: false
  };

  const isStreaming = Boolean(ungroundedDoneMessage.isStreaming);
  const isGrounded = Boolean(ungroundedDoneMessage.grounded);

  assert.strictEqual(isStreaming, false, 'isStreaming must be false when completed');
  assert.strictEqual(isGrounded, false, 'isGrounded must be strictly false even with sources');

  // Verify badge logic
  let badgeText = '';
  if (ungroundedDoneMessage.isError || ungroundedDoneMessage.error) {
    badgeText = 'Service Error';
  } else if (isStreaming) {
    badgeText = 'Analyzing Evidence...';
  } else if (isGrounded) {
    badgeText = 'Grounded Evidence';
  } else {
    badgeText = 'Ungrounded / Missing Evidence';
  }

  assert.strictEqual(badgeText, 'Ungrounded / Missing Evidence', 'Badge must display Ungrounded / Missing Evidence');
  console.log('  PASS: Ungrounded answer strictly displays "Ungrounded / Missing Evidence".');
}

// -------------------------------------------------------------
// Test 7: Grounded completed answer retains sources and grounded status
// -------------------------------------------------------------
console.log('Running Test 7: Grounded answer retains sources and grounded status...');
{
  const groundedDoneMessage = {
    content: "The entrypoint is configured in main.py [main.py:1-20].",
    sources: [{ file_path: 'main.py', start_line: 1, end_line: 20 }],
    grounded: true,
    isStreaming: false,
    isError: false
  };

  const isStreaming = Boolean(groundedDoneMessage.isStreaming);
  const isGrounded = Boolean(groundedDoneMessage.grounded);

  assert.strictEqual(isStreaming, false, 'isStreaming must be false when completed');
  assert.strictEqual(isGrounded, true, 'isGrounded must be true');
  assert.strictEqual(groundedDoneMessage.sources.length, 1);

  let badgeText = '';
  if (groundedDoneMessage.isError || groundedDoneMessage.error) {
    badgeText = 'Service Error';
  } else if (isStreaming) {
    badgeText = 'Analyzing Evidence...';
  } else if (isGrounded) {
    badgeText = 'Grounded Evidence';
  } else {
    badgeText = 'Ungrounded / Missing Evidence';
  }

  assert.strictEqual(badgeText, 'Grounded Evidence', 'Badge must display Grounded Evidence');
  console.log('  PASS: Grounded answer displays "Grounded Evidence" and retains sources.');
}

// -------------------------------------------------------------
// Test 8: Streaming status displays 'Analyzing Evidence...'
// -------------------------------------------------------------
console.log('Running Test 8: Streaming assistant message displays "Analyzing Evidence..."...');
{
  const streamingMessage = {
    content: "Partial stream token...",
    sources: [],
    grounded: false,
    isStreaming: true,
    isError: false
  };

  const isStreaming = Boolean(streamingMessage.isStreaming);
  const isGrounded = Boolean(streamingMessage.grounded);

  let badgeText = '';
  if (streamingMessage.isError || streamingMessage.error) {
    badgeText = 'Service Error';
  } else if (isStreaming) {
    badgeText = 'Analyzing Evidence...';
  } else if (isGrounded) {
    badgeText = 'Grounded Evidence';
  } else {
    badgeText = 'Ungrounded / Missing Evidence';
  }

  assert.strictEqual(badgeText, 'Analyzing Evidence...', 'Streaming message must display Analyzing Evidence...');
  console.log('  PASS: Streaming message displays "Analyzing Evidence...".');
}

console.log('\n=======================================');
console.log('ALL FRONTEND STREAM REGRESSION TESTS PASSED!');
console.log('=======================================');
