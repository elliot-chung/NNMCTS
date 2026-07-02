export function maskedSoftmax(
  logits: Float32Array | number[],
  mask: Float32Array | number[],
): Float32Array {
  const size = logits.length;
  const result = new Float32Array(size);

  let maskSum = 0;
  for (let i = 0; i < size; i++) {
    maskSum += mask[i];
  }

  if (maskSum === 0) {
    for (let i = 0; i < size; i++) {
      result[i] = mask[i];
    }
    return result;
  }

  let maxLogit = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < size; i++) {
    if (mask[i] !== 0) {
      maxLogit = Math.max(maxLogit, logits[i]);
    }
  }

  let sum = 0;
  for (let i = 0; i < size; i++) {
    if (mask[i] === 0) {
      result[i] = 0;
      continue;
    }
    const exp = Math.exp(logits[i] - maxLogit);
    result[i] = exp;
    sum += exp;
  }

  for (let i = 0; i < size; i++) {
    if (mask[i] !== 0) {
      result[i] /= sum;
    }
  }

  return result;
}
