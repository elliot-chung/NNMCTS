This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Neural model

The browser ONNX model (`public/models/uttt-v1.onnx`) is exported from the local checkpoint:

`artifacts/gpu-20260701-192839/checkpoints/round_020.pt`

Re-export after updating the checkpoint:

```bash
python scripts/export_onnx.py
python scripts/validate_onnx.py
python scripts/generate_uttt_fixtures.py
```
