# Self-Supervised and Contrastive Representation Learning on STL-10

## Project summary

This project investigates whether unlabeled images can improve image representations when only 10% of STL-10's labeled training data is available. I used the same randomly initialized ResNet-18 backbone for all experiments and compared:

1. Fully supervised training on 500 labeled images.
2. Rotation self-supervised pretraining followed by frozen linear evaluation.
3. SimCLR-style contrastive pretraining followed by frozen linear evaluation.

The key result is that SimCLR learned the most transferable representation: **53.20% test accuracy** using only a linear classifier trained on the same 500 labeled images. It also achieved the best nearest-neighbor precision.

## Results

| Part | Method | Pretraining | Linear/final train accuracy | Test accuracy |
|---|---|---:|---:|---:|
| A | Supervised ResNet-18 on 500 labels | — | 0.7120 | **0.3639** |
| B | Rotation SSL + frozen linear classifier | 15 epochs, 100,000 unlabeled images | 0.3600 | **0.3763** |
| C | SimCLR + frozen linear classifier | 20 epochs, 20,000 unlabeled images | 0.7040 | **0.5320** |

The reported Part B result is the rerun with linear-classifier learning rate `1e-2`. The earlier `1e-3` probe produced `0.1921` test accuracy because its loss remained close to the random-guessing level; that run is preserved in `RUN_LOG.txt` as a troubleshooting result, not used as the final Part B comparison.

### Nearest-neighbor retrieval

For every test image, I extracted a 512-dimensional embedding, L2-normalized it, and retrieved the five nearest other test images using cosine similarity.

| Encoder | Top-5 neighbor precision |
|---|---:|
| Supervised | 0.3670 |
| Rotation SSL | 0.3352 |
| SimCLR | **0.4819** |

Random retrieval would have an expected precision of approximately `0.10` because STL-10 has 10 classes.

## Dataset and experimental design

STL-10 contains 96×96 RGB images from 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, and truck.

| Split | Images | Use |
|---|---:|---|
| Labeled train | 5,000 | A balanced 500-image subset: 50 per class |
| Test | 8,000 | Final accuracy and nearest-neighbor analysis |
| Unlabeled | 100,000 | Rotation pretraining; a 20,000-image subset for SimCLR |

The same balanced 500-image labeled subset was used in Parts A, B, and C. The subset was selected with seed `2122` and saved as `subset_idx.npy`.

All backbones were `torchvision.models.resnet18(weights=None)`. For Parts B and C, the final classification layer was replaced by `Identity`, producing a 512-dimensional encoder representation.

## ResNet-18 and training-loop design

ResNet-18 is a convolutional neural network built from residual blocks. Each block learns a transformation of its input and adds the original input through a skip connection. This makes optimization more stable than learning every layer as a completely new transformation. The final fully connected layer normally maps the learned representation to class logits; in Parts B and C it was removed so the network could be used as an encoder.

The training loop follows the same basic pattern in every experiment:

1. Load a batch and move it to the GPU.
2. Apply the relevant augmentation or generate the self-supervised target.
3. Run a forward pass through the model.
4. Compute the loss.
5. Backpropagate the loss and update only the intended trainable parameters.
6. Measure training and test performance at the end of the epoch.

For linear evaluation, the encoder was placed in evaluation mode, all encoder parameters had `requires_grad=False`, and only the 5,130-parameter linear classifier was optimized.

## Part A — Supervised limited-label baseline

The entire ResNet-18 was trained end-to-end on the 500 labeled images for 15 epochs.

- Augmentation: random crop with padding 8 and random horizontal flip.
- Optimizer: Adam.
- Learning rate: `1e-3`.
- Weight decay: `1e-4`.
- Batch size: `64`.
- Final train accuracy: `71.20%`.
- Final test accuracy: **`36.39%`**.

The training accuracy rose much faster than test accuracy. This gap indicates overfitting: with only 500 labels, the model learned the training examples more easily than general visual features.

## Part B — Rotation self-supervised learning

### Pretraining task

Each unlabeled image was transformed into four examples by applying rotations of 0°, 90°, 180°, and 270°. A temporary `Linear(512, 4)` head predicted the rotation. The rotation labels were generated automatically, so no human class labels were used.

- Data: all 100,000 unlabeled images.
- Epochs: 15.
- Rotation accuracy: `71.7%` → `90.5%`.
- After pretraining, the rotation head was removed.

### Frozen linear evaluation

The encoder was frozen and only a new `Linear(512, 10)` classifier was trained on the same 500 labeled images for 20 epochs. Only `5,130` parameters were trainable, and the check `encoder unchanged: True` confirmed that the encoder did not update.

- Final train accuracy: `36.00%`.
- Final test accuracy: **`37.63%`**.

An earlier probe with learning rate `1e-3` reached only `19.21%`. Its loss decreased from `2.314` to `2.162`, so the linear classifier had barely learned. The rerun used `1e-2`, selected from the training behavior rather than from test accuracy.

## Part C — SimCLR-style contrastive learning

### Pretraining task

For each unlabeled image, two independently augmented views were created. The encoder and a projection MLP were trained to make the two views similar and views from different images dissimilar.

Each view used the following four augmentations:

- Random resized crop, scale `0.2–1.0`.
- Random horizontal flip.
- Color jitter with probability `0.8`.
- Random grayscale with probability `0.2`.

The projection head was `512 → 512 → 128`. The contrastive loss was applied to the 128-dimensional projected features, while the projection head was discarded after pretraining.

### NT-Xent details

The projected features were L2-normalized, so their dot product was cosine similarity. With batch size `256`, there were `512` views; each view had one positive partner and `511` candidates. The temperature was `τ = 0.2`.

The contrastive loss decreased from `5.364` to `2.882`, while the positive-pair top-1 rate increased from `4.3%` to `73.8%`.

### Frozen linear evaluation

The projection head was removed. The 512-dimensional encoder was frozen, and only a `Linear(512, 10)` classifier was trained on the same 500 labeled images for 20 epochs.

- Final train accuracy: `70.40%`.
- Final test accuracy: **`53.20%`**.
- Encoder unchanged: `True`.

## Why Part C won even though its pretext accuracy was lower

The `90.5%` rotation accuracy and `73.8%` SimCLR positive-pair accuracy measure different tasks, so they cannot be compared directly.

- Rotation is a 4-way classification problem. A model can often solve it using orientation shortcuts such as sky, ground, horizon, or object placement.
- SimCLR asks the model to identify the matching view among `511` candidates. Its positive-pair accuracy is therefore measured on a much harder retrieval problem.
- The downstream frozen linear evaluation is the fair comparison. SimCLR achieved `53.20%`, versus `37.63%` for rotation and `36.39%` for supervised training.

SimCLR likely transferred better because the two augmented views preserve object identity while changing appearance, crop, color, and sometimes grayscale. The encoder therefore has an incentive to learn stable object and shape information instead of only orientation cues.

## Part D — Nearest-neighbor visualization

The same four query images were used for all three encoders. For each query, the query image and its top five cosine-similarity neighbors were displayed with class labels. Green labels indicate the same class as the query; red labels indicate a different class.

The figures are saved as:

- `figures/partD_query1.png`
- `figures/partD_query2.png`
- `figures/partD_query3.png`
- `figures/partD_query4.png`

The visualizations showed that:

- SimCLR produced the cleanest neighbors overall, especially for the truck query.
- Supervised and rotation features often captured broad visual groups, such as vehicles versus animals, but confused fine-grained classes.
- The monkey query was difficult for all three encoders, showing that representation quality was not perfect even when the overall retrieval metric was strong.
- Some rotation neighbors appeared to match background, layout, or color more than object identity, consistent with shortcut learning.
- The all-test-set precision numbers are stronger evidence than four individual query panels, but the panels make the types of errors easier to interpret.

## Interpretation and possible improvements

### Supervised baseline

The supervised model learned directly from only 500 labels and overfit. Improvements could include stronger augmentation, label smoothing, a learning-rate schedule, higher weight decay, a smaller model, and early stopping selected using a validation split.

### Rotation SSL

Rotation pretraining learned the pretext task well but transferred less effectively. Improvements could include longer pretraining, a learning-rate schedule, training the linear probe to convergence, using intermediate-layer features, and augmentations that reduce orientation shortcuts.

### SimCLR

SimCLR could improve with all 100,000 unlabeled images, longer pretraining, a larger effective batch size, a learning-rate warmup/cosine schedule, stronger augmentation, and multiple random seeds. A larger batch would provide more negative examples per step.

### Experimental limitations

- Results come from one main random seed; small differences should not be overinterpreted.
- Rotation used 100,000 unlabeled images, while SimCLR used 20,000, so the pretraining budgets were not identical.
- The Part B learning rate was changed after observing a flat training run. A validation split would provide a cleaner hyperparameter-selection procedure.
- Part B's probe loss was still decreasing at epoch 20, so additional probe training might improve it.
- The nearest-neighbor query panels use only four queries; the full 8,000-image precision is the more reliable aggregate metric.

## Interview explanation

> I trained the same ResNet-18 from scratch under three label-efficient settings on STL-10. A supervised model trained on 500 labels reached 36.4% test accuracy but overfit. Rotation pretraining followed by a frozen linear probe reached 37.6%. SimCLR learned from 20,000 unlabeled images using two augmented views and NT-Xent contrastive loss; after freezing the encoder, the same 500 labels produced 53.2% test accuracy. The nearest-neighbor precision also favored SimCLR, which suggests its embeddings captured more stable object-level similarity rather than only color, background, or orientation.

### Interview questions and concise answers

**What is self-supervised learning?**

Self-supervised learning creates training targets from the input data itself, so the model can learn from unlabeled examples. In this project, rotation prediction and matching two augmented views supplied the supervision.

**Why freeze the encoder during linear evaluation?**

Freezing isolates representation quality. Since only the linear classifier learns from the 500 labels, the result measures how linearly separable the learned features already are rather than allowing the entire network to adapt to the labels.

**Why use a projection head in SimCLR?**

The projection head gives the contrastive objective a task-specific space. After pretraining, it is discarded and the encoder representation is used because it is generally more transferable to downstream tasks.

**What does the temperature do?**

Temperature scales the cosine-similarity logits in NT-Xent. A lower temperature makes the distribution sharper and penalizes confusing hard negatives more strongly. This experiment used `τ = 0.2`.

**Why does batch size matter in SimCLR?**

Other images in the batch act as negative examples. A larger batch provides more negatives and makes the matching task more informative, although it also increases memory and computation.

**How did you verify that the encoder stayed frozen?**

The training loop reported only 5,130 trainable parameters, corresponding to `512 × 10 + 10` in the linear classifier. It also compared an encoder fingerprint before and after training and printed `encoder unchanged: True`.

**What would you do next?**

I would pretrain SimCLR on all 100,000 unlabeled images, train for more epochs, use a larger effective batch size, add a learning-rate schedule, and repeat every experiment across multiple seeds with mean and standard deviation.

## Reproducibility files

| File | Purpose |
|---|---|
| `Training_CNN.ipynb` | Executed notebook and implementation |
| `RUN_LOG.txt` | Console output for Parts A–C and Part D precision |
| `METRICS.md` | Compact metrics table |
| `results.json` | Machine-readable results |
| `AI_USE.md` | AI-use disclosure |
| `subset_idx.npy` | Reproducible 500-image labeled subset |
| `figures/` | Part A curve and Part D visualizations |
