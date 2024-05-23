ControlLDM(
  (model): DiffusionWrapper(
    (diffusion_model): StableVITON(
      (time_embed): Sequential(
        (0): Linear(in_features=320, out_features=1280, bias=True)
        (1): SiLU()
        (2): Linear(in_features=1280, out_features=1280, bias=True)
      )
      (input_blocks): ModuleList(
        (0): TimestepEmbedSequential(
          (0): Conv2d(13, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (1-2): 2 x TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=320, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Identity()
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
            (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=320, out_features=320, bias=False)
                  (to_k): Linear(in_features=320, out_features=320, bias=False)
                  (to_v): Linear(in_features=320, out_features=320, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=320, out_features=320, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=320, out_features=2560, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=1280, out_features=320, bias=True)
                  )
                )
                (attn2): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=320, out_features=320, bias=False)
                  (to_k): Linear(in_features=768, out_features=320, bias=False)
                  (to_v): Linear(in_features=768, out_features=320, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=320, out_features=320, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (3): TimestepEmbedSequential(
          (0): Downsample(
            (op): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          )
        )
        (4): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=640, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
            (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=640, out_features=640, bias=False)
                  (to_v): Linear(in_features=640, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=640, out_features=5120, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=2560, out_features=640, bias=True)
                  )
                )
                (attn2): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=768, out_features=640, bias=False)
                  (to_v): Linear(in_features=768, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (5): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=640, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Identity()
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
            (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=640, out_features=640, bias=False)
                  (to_v): Linear(in_features=640, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=640, out_features=5120, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=2560, out_features=640, bias=True)
                  )
                )
                (attn2): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=768, out_features=640, bias=False)
                  (to_v): Linear(in_features=768, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (6): TimestepEmbedSequential(
          (0): Downsample(
            (op): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          )
        )
        (7): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
            (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=1280, out_features=10240, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=5120, out_features=1280, bias=True)
                  )
                )
                (attn2): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=768, out_features=1280, bias=False)
                  (to_v): Linear(in_features=768, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (8): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Identity()
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
            (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=1280, out_features=10240, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=5120, out_features=1280, bias=True)
                  )
                )
                (attn2): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=768, out_features=1280, bias=False)
                  (to_v): Linear(in_features=768, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (9): TimestepEmbedSequential(
          (0): Downsample(
            (op): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          )
        )
        (10-11): 2 x TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Identity()
          )
        )
      )
      (middle_block): TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Identity()
        )
        (1): SpatialTransformer(
          (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
          (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=1280, out_features=10240, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=5120, out_features=1280, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=768, out_features=1280, bias=False)
                (to_v): Linear(in_features=768, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Identity()
        )
      )
      (output_blocks): ModuleList(
        (0-1): 2 x TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (2): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): Upsample(
            (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (3-4): 2 x TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
            (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=1280, out_features=10240, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=5120, out_features=1280, bias=True)
                  )
                )
                (attn2): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=768, out_features=1280, bias=False)
                  (to_v): Linear(in_features=768, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (5): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 1920, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
            (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=1280, out_features=10240, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=5120, out_features=1280, bias=True)
                  )
                )
                (attn2): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=768, out_features=1280, bias=False)
                  (to_v): Linear(in_features=768, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
          (2): Upsample(
            (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (6): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 1920, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=640, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(1920, 640, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
            (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=640, out_features=640, bias=False)
                  (to_v): Linear(in_features=640, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=640, out_features=5120, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=2560, out_features=640, bias=True)
                  )
                )
                (attn2): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=768, out_features=640, bias=False)
                  (to_v): Linear(in_features=768, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (7): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=640, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
            (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=640, out_features=640, bias=False)
                  (to_v): Linear(in_features=640, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=640, out_features=5120, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=2560, out_features=640, bias=True)
                  )
                )
                (attn2): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=768, out_features=640, bias=False)
                  (to_v): Linear(in_features=768, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (8): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 960, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=640, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
            (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=640, out_features=640, bias=False)
                  (to_v): Linear(in_features=640, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=640, out_features=5120, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=2560, out_features=640, bias=True)
                  )
                )
                (attn2): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=768, out_features=640, bias=False)
                  (to_v): Linear(in_features=768, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
          )
          (2): Upsample(
            (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (9): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 960, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(960, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=320, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
            (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=320, out_features=320, bias=False)
                  (to_k): Linear(in_features=320, out_features=320, bias=False)
                  (to_v): Linear(in_features=320, out_features=320, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=320, out_features=320, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=320, out_features=2560, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=1280, out_features=320, bias=True)
                  )
                )
                (attn2): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=320, out_features=320, bias=False)
                  (to_k): Linear(in_features=768, out_features=320, bias=False)
                  (to_v): Linear(in_features=768, out_features=320, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=320, out_features=320, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (10-11): 2 x TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=320, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
            (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=320, out_features=320, bias=False)
                  (to_k): Linear(in_features=320, out_features=320, bias=False)
                  (to_v): Linear(in_features=320, out_features=320, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=320, out_features=320, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=320, out_features=2560, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=1280, out_features=320, bias=True)
                  )
                )
                (attn2): MemoryEfficientCrossAttention(
                  (to_q): Linear(in_features=320, out_features=320, bias=False)
                  (to_k): Linear(in_features=768, out_features=320, bias=False)
                  (to_v): Linear(in_features=768, out_features=320, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=320, out_features=320, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
          )
        )
      )
      (out): Sequential(
        (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (warp_flow_blks): ModuleList(
        (0-1): 2 x CustomSpatialTransformer(
          (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
          (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): CustomBasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=1280, out_features=10240, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=5120, out_features=1280, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (2-3): 2 x CustomSpatialTransformer(
          (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
          (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): CustomBasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=1280, out_features=10240, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=5120, out_features=1280, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=640, out_features=1280, bias=False)
                (to_v): Linear(in_features=640, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (4): CustomSpatialTransformer(
          (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
          (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): CustomBasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=640, out_features=640, bias=False)
                (to_v): Linear(in_features=640, out_features=640, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=640, out_features=5120, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=2560, out_features=640, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=640, out_features=640, bias=False)
                (to_v): Linear(in_features=640, out_features=640, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (5-6): 2 x CustomSpatialTransformer(
          (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
          (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): CustomBasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=640, out_features=640, bias=False)
                (to_v): Linear(in_features=640, out_features=640, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=640, out_features=5120, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=2560, out_features=640, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=320, out_features=640, bias=False)
                (to_v): Linear(in_features=320, out_features=640, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (7-8): 2 x CustomSpatialTransformer(
          (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
          (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): CustomBasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=320, out_features=320, bias=False)
                (to_v): Linear(in_features=320, out_features=320, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=320, out_features=2560, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=1280, out_features=320, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=320, out_features=320, bias=False)
                (to_v): Linear(in_features=320, out_features=320, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (warp_zero_convs): ModuleList(
        (0-3): 4 x Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        (4-6): 3 x Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        (7-8): 2 x Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
  (imagenet_norm): Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
  (first_stage_model): AutoencoderKL(
    (encoder): Encoder(
      (conv_in): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (down): ModuleList(
        (0): Module(
          (block): ModuleList(
            (0-1): 2 x ResnetBlock(
              (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
              (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (downsample): Downsample(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
          )
        )
        (1): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
              (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (nin_shortcut): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (downsample): Downsample(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
          )
        )
        (2): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (nin_shortcut): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (downsample): Downsample(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2))
          )
        )
        (3): Module(
          (block): ModuleList(
            (0-1): 2 x ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
        )
      )
      (mid): Module(
        (block_1): ResnetBlock(
          (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (attn_1): MemoryEfficientAttnBlock(
          (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
          (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        )
        (block_2): ResnetBlock(
          (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (norm_out): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv_out): Conv2d(512, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (decoder): Decoder(
      (conv_in): Conv2d(4, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (mid): Module(
        (block_1): ResnetBlock(
          (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (attn_1): MemoryEfficientAttnBlock(
          (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
          (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        )
        (block_2): ResnetBlock(
          (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (up): ModuleList(
        (0): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (nin_shortcut): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
            )
            (1-2): 2 x ResnetBlock(
              (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
              (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
        )
        (1): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (nin_shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
            )
            (1-2): 2 x ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (upsample): Upsample(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (2-3): 2 x Module(
          (block): ModuleList(
            (0-2): 3 x ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (upsample): Upsample(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
      )
      (norm_out): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv_out): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (loss): Identity()
    (quant_conv): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
    (post_quant_conv): Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1))
  )
  (cond_stage_model): FrozenCLIPImageEmbedder(
    (transformer): CLIPVisionModel(
      (vision_model): CLIPVisionTransformer(
        (embeddings): CLIPVisionEmbeddings(
          (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
          (position_embedding): Embedding(257, 1024)
        )
        (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (encoder): CLIPEncoder(
          (layers): ModuleList(
            (0-23): 24 x CLIPEncoderLayer(
              (self_attn): CLIPAttention(
                (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
                (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
                (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
                (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
              )
              (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): CLIPMLP(
                (activation_fn): QuickGELUActivation()
                (fc1): Linear(in_features=1024, out_features=4096, bias=True)
                (fc2): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            )
          )
        )
        (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
    (final_ln): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    (mapper): Transformer(
      (resblocks): ModuleList(
        (0-4): 5 x ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (c_qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (c_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (attention): QKVMultiheadAttention()
          )
          (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (mlp): MLP(
            (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
            (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
            (gelu): GELU(approximate='none')
          )
          (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
  (proj_out): Linear(in_features=1024, out_features=768, bias=True)
  (lastzc): Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1))
  (control_model): WarpingControlNet(
    (time_embed): Sequential(
      (0): Linear(in_features=320, out_features=1280, bias=True)
      (1): SiLU()
      (2): Linear(in_features=1280, out_features=1280, bias=True)
    )
    (input_blocks): ModuleList(
      (0): TimestepEmbedSequential(
        (0): Conv2d(13, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (1-2): 2 x TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=320, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Identity()
        )
        (1): SpatialTransformer(
          (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
          (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=320, out_features=320, bias=False)
                (to_v): Linear(in_features=320, out_features=320, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=320, out_features=2560, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=1280, out_features=320, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=768, out_features=320, bias=False)
                (to_v): Linear(in_features=768, out_features=320, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (3): TimestepEmbedSequential(
        (0): Downsample(
          (op): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
      (4): TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=640, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): SpatialTransformer(
          (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
          (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=640, out_features=640, bias=False)
                (to_v): Linear(in_features=640, out_features=640, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=640, out_features=5120, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=2560, out_features=640, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=768, out_features=640, bias=False)
                (to_v): Linear(in_features=768, out_features=640, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (5): TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=640, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Identity()
        )
        (1): SpatialTransformer(
          (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
          (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=640, out_features=640, bias=False)
                (to_v): Linear(in_features=640, out_features=640, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=640, out_features=5120, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=2560, out_features=640, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=768, out_features=640, bias=False)
                (to_v): Linear(in_features=768, out_features=640, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (6): TimestepEmbedSequential(
        (0): Downsample(
          (op): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
      (7): TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): SpatialTransformer(
          (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
          (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=1280, out_features=10240, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=5120, out_features=1280, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=768, out_features=1280, bias=False)
                (to_v): Linear(in_features=768, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (8): TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Identity()
        )
        (1): SpatialTransformer(
          (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
          (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=1280, out_features=10240, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=5120, out_features=1280, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=768, out_features=1280, bias=False)
                (to_v): Linear(in_features=768, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (9): TimestepEmbedSequential(
        (0): Downsample(
          (op): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
      (10-11): 2 x TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Identity()
        )
      )
    )
    (middle_block): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
      (1): SpatialTransformer(
        (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (attn1): MemoryEfficientCrossAttention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=1280, out_features=1280, bias=False)
              (to_v): Linear(in_features=1280, out_features=1280, bias=False)
              (to_out): Sequential(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (ff): FeedForward(
              (net): Sequential(
                (0): GEGLU(
                  (proj): Linear(in_features=1280, out_features=10240, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=5120, out_features=1280, bias=True)
              )
            )
            (attn2): MemoryEfficientCrossAttention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=768, out_features=1280, bias=False)
              (to_v): Linear(in_features=768, out_features=1280, bias=False)
              (to_out): Sequential(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          )
        )
        (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (output_blocks): ModuleList(
      (0-1): 2 x TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (2): TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): Upsample(
          (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (3-4): 2 x TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): SpatialTransformer(
          (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
          (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=1280, out_features=10240, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=5120, out_features=1280, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=768, out_features=1280, bias=False)
                (to_v): Linear(in_features=768, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (5): TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 1920, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Conv2d(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): SpatialTransformer(
          (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
          (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=1280, out_features=10240, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=5120, out_features=1280, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=768, out_features=1280, bias=False)
                (to_v): Linear(in_features=768, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): Upsample(
          (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (6): TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 1920, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=640, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Conv2d(1920, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): SpatialTransformer(
          (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
          (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=640, out_features=640, bias=False)
                (to_v): Linear(in_features=640, out_features=640, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=640, out_features=5120, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=2560, out_features=640, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=768, out_features=640, bias=False)
                (to_v): Linear(in_features=768, out_features=640, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (7): TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=640, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): SpatialTransformer(
          (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
          (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=640, out_features=640, bias=False)
                (to_v): Linear(in_features=640, out_features=640, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=640, out_features=5120, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=2560, out_features=640, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=768, out_features=640, bias=False)
                (to_v): Linear(in_features=768, out_features=640, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (8): TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 960, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=640, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): SpatialTransformer(
          (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
          (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=640, out_features=640, bias=False)
                (to_v): Linear(in_features=640, out_features=640, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=640, out_features=5120, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=2560, out_features=640, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=768, out_features=640, bias=False)
                (to_v): Linear(in_features=768, out_features=640, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): Upsample(
          (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (9): TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 960, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(960, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=320, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): SpatialTransformer(
          (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
          (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=320, out_features=320, bias=False)
                (to_v): Linear(in_features=320, out_features=320, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=320, out_features=2560, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=1280, out_features=320, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=768, out_features=320, bias=False)
                (to_v): Linear(in_features=768, out_features=320, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (10-11): 2 x TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=320, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): SpatialTransformer(
          (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
          (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=320, out_features=320, bias=False)
                (to_v): Linear(in_features=320, out_features=320, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=320, out_features=2560, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=1280, out_features=320, bias=True)
                )
              )
              (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=768, out_features=320, bias=False)
                (to_v): Linear(in_features=768, out_features=320, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    (out): Sequential(
      (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
      (1): SiLU()
      (2): Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (cond_first_block): TimestepEmbedSequential(
      (0): Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
)