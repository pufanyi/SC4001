# SC4001

```sh
python tools/export_fsdp2_to_hf.py --checkpoint-dir outputs/convnextv2-huge-22k-384/step_200 --repo-id pufanyi/SC4001-convnextv2-huge-22k-384-wsd-adamw
```

```sh
python eval/pipeline/run.py \
    --model /mnt/umm/users/pufanyi/workspace/qwen/outputs/convnextv2-huge-22k-384/step_200/hf_export \
    --dataset pufanyi/flowers102 \
    --split test \
    --batch-size 64 \
    --metrics-output results.json
```
