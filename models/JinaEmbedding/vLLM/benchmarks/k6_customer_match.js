import http from 'k6/http';
import { check } from 'k6';
import { Trend, Counter, Rate } from 'k6/metrics';

const MODE = __ENV.MODE || 'concurrency'; // 'concurrency', 'multi_rps', 'dedicated_rps'
const HTTP_URL = __ENV.HTTP_URL || 'http://jina-embedding-service:8000/prompt_c2';
const DURATION = __ENV.DURATION || '12s';
const VUS = parseInt(__ENV.VUS || '1', 10);
const RPS = parseInt(__ENV.RPS || '50', 10);
const PAYLOAD_KB = parseInt(__ENV.PAYLOAD_KB || '1', 10);

// Custom metrics per payload size
const lat_1kb = new Trend('lat_1kb', true);
const lat_2kb = new Trend('lat_2kb', true);
const lat_3kb = new Trend('lat_3kb', true);
const lat_4kb = new Trend('lat_4kb', true);
const lat_5kb = new Trend('lat_5kb', true);
const lat_7kb = new Trend('lat_7kb', true);

const err_1kb = new Rate('err_1kb');
const err_2kb = new Rate('err_2kb');
const err_3kb = new Rate('err_3kb');
const err_4kb = new Rate('err_4kb');
const err_5kb = new Rate('err_5kb');
const err_7kb = new Rate('err_7kb');

function makeRandomChars(length) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789     ';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt((i * 31 + 17) % chars.length);
  }
  return result;
}

const PAYLOADS = {
  1: JSON.stringify({ text: makeRandomChars(1024) }),
  2: JSON.stringify({ text: makeRandomChars(2048) }),
  3: JSON.stringify({ text: makeRandomChars(3072) }),
  4: JSON.stringify({ text: makeRandomChars(4096) }),
  5: JSON.stringify({ text: makeRandomChars(5120) }),
  7: JSON.stringify({ text: makeRandomChars(7168) }),
};

const MULTI_SIZES = [1, 2, 5, 7];

export const options = (function () {
  if (MODE === 'concurrency') {
    return {
      scenarios: {
        closed_loop: {
          executor: 'constant-vus',
          vus: VUS,
          duration: DURATION,
        },
      },
    };
  } else {
    return {
      scenarios: {
        open_loop: {
          executor: 'constant-arrival-rate',
          rate: RPS,
          timeUnit: '1s',
          duration: DURATION,
          preAllocatedVUs: Math.min(Math.max(RPS * 2, 50), 300),
          maxVUs: 400,
        },
      },
    };
  }
})();

const params = {
  headers: { 'Content-Type': 'application/json' },
  timeout: '15s',
};

export default function () {
  let kb = PAYLOAD_KB;
  if (MODE === 'multi_rps') {
    kb = MULTI_SIZES[__ITER % MULTI_SIZES.length];
  }
  const body = PAYLOADS[kb];
  const res = http.post(HTTP_URL, body, params);
  const ok = res.status === 200;
  const dur = res.timings.duration;

  if (kb === 1) { lat_1kb.add(dur); err_1kb.add(!ok); }
  else if (kb === 2) { lat_2kb.add(dur); err_2kb.add(!ok); }
  else if (kb === 3) { lat_3kb.add(dur); err_3kb.add(!ok); }
  else if (kb === 4) { lat_4kb.add(dur); err_4kb.add(!ok); }
  else if (kb === 5) { lat_5kb.add(dur); err_5kb.add(!ok); }
  else if (kb === 7) { lat_7kb.add(dur); err_7kb.add(!ok); }

  check(res, { 'status is 200': (r) => r.status === 200 });
}
