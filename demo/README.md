This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Neural model

The browser ONNX model (`public/models/uttt-v1.onnx`) is exported from the local checkpoint:

`artifacts/gpu-20260701-192839/checkpoints/round_020.pt`

Re-export after updating the checkpoint:

```bash
python scripts/export_onnx.py --checkpoint artifacts/<run-id>/round_020.pt
# copy to demo/public/models/ if you used a custom --output path
python scripts/validate_onnx.py --checkpoint artifacts/<run-id>/round_020.pt
```
