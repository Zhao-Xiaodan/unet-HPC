# HPC Sync Checklist - Microbead Optimized Training

## Files to Sync to HPC

### ✅ Required Files (Must Upload)

```bash
# Core training files
train_microbead_optimized.py          # Optimized training script
pbs_microbead_optimized.sh            # PBS job submission script
models.py or 224_225_226_models.py    # Model definitions (already on HPC)

# Dataset (should already be on HPC)
dataset_microscope/images/            # Your microscope images
dataset_microscope/masks/             # Corresponding masks
```

### 📊 Recommended Files (Optional but Helpful)

```bash
# Analysis and documentation
MICROBEAD_ANALYSIS_RESULTS.md         # Analysis findings
DOMAIN_SHIFT_ANALYSIS.md              # Theoretical explanation
dataset_analysis/                     # Analysis outputs (if generated locally)
```

---

## Step-by-Step Sync Instructions

### Option 1: Git Sync (Recommended if using Git)

```bash
# On local machine (in unet-HPC directory)
git add train_microbead_optimized.py
git add pbs_microbead_optimized.sh
git add MICROBEAD_ANALYSIS_RESULTS.md
git add DOMAIN_SHIFT_ANALYSIS.md
git commit -m "Add optimized microbead training with corrected hyperparameters

- LR=1e-4 (was 1e-3) for 36× more dense objects
- Batch size=32 (was 8-16) for gradient stability
- Dropout=0.3 for regularization
- Dice loss instead of Focal loss
- Stratified train/val split by object density"

git push

# On HPC
ssh phyzxi@hpc
cd ~/scratch/unet-HPC
git pull
```

### Option 2: Direct SCP (If not using Git)

```bash
# From local machine
scp train_microbead_optimized.py phyzxi@hpc:~/scratch/unet-HPC/
scp pbs_microbead_optimized.sh phyzxi@hpc:~/scratch/unet-HPC/
scp MICROBEAD_ANALYSIS_RESULTS.md phyzxi@hpc:~/scratch/unet-HPC/
scp DOMAIN_SHIFT_ANALYSIS.md phyzxi@hpc:~/scratch/unet-HPC/

# Optional: Sync analysis results
scp -r dataset_analysis/ phyzxi@hpc:~/scratch/unet-HPC/
```

---

## Pre-Submission Verification (On HPC)

```bash
# SSH to HPC
ssh phyzxi@hpc
cd ~/scratch/unet-HPC

# 1. Verify files are present
ls -lh train_microbead_optimized.py pbs_microbead_optimized.sh models.py

# 2. Verify dataset exists
ls dataset_microscope/images/ | head -5
ls dataset_microscope/masks/ | head -5

# 3. Check file counts
echo "Images: $(find dataset_microscope/images/ -type f | wc -l)"
echo "Masks: $(find dataset_microscope/masks/ -type f | wc -l)"

# 4. Make PBS script executable (if needed)
chmod +x pbs_microbead_optimized.sh

# 5. Quick syntax check
head -20 train_microbead_optimized.py
head -20 pbs_microbead_optimized.sh
```

---

## Submit Job

```bash
# Submit the optimized training job
qsub pbs_microbead_optimized.sh

# You should see output like:
# 278XXX.stdct-mgmt-02

# Check job status
qstat -u phyzxi

# Monitor job output
tail -f Microbead_Optimized_Training.o<JOBID>
```

---

## Expected Job Output

### During Training

```
=== OPTIMIZED HYPERPARAMETERS ===
Learning Rate:
  Mitochondria: 1e-3 (UNet), 1e-4 (Attention)
  Microbeads:   1e-4 (all models) ← 10× LOWER

Batch Size:
  Mitochondria: 8-16
  Microbeads:   32 ← 2-4× LARGER

...

🚀 STARTING OPTIMIZED MICROBEAD TRAINING
Training 3 models...

Model 1/3: STANDARD U-NET
Epoch 1/100
  - loss: 0.3521 - accuracy: 0.8934 - jacard_coef: 0.3421 - val_loss: 0.3012 - val_jacard_coef: 0.4123
Epoch 2/100
  - loss: 0.2987 - accuracy: 0.9012 - jacard_coef: 0.4234 - val_loss: 0.2765 - val_jacard_coef: 0.4556
...
```

### Success Indicators

✅ **Val Jaccard increases steadily** (not spiking at epoch 1)
✅ **Val Jaccard reaches > 0.30** by epoch 10
✅ **Val Jaccard reaches > 0.50** by epoch 20-40
✅ **No collapse to 0.0** in later epochs
✅ **Training completes** with stable metrics

### Warning Signs

⚠️ **Val Jaccard peaks at epoch 1-5 then drops** → LR still too high
⚠️ **Val Jaccard oscillates wildly** → Batch size too small
⚠️ **Val Jaccard stuck < 0.20** → Check dataset or loss function

---

## Post-Training Checks

### On HPC (After Job Completes)

```bash
# 1. Find output directory
ls -ltd microbead_training_*

# Most recent directory
OUTPUT=$(ls -td microbead_training_* | head -1)
echo "Latest training: $OUTPUT"

# 2. Check generated files
ls -lh $OUTPUT/

# Expected files:
# - best_unet_model.hdf5
# - best_attention_unet_model.hdf5
# - best_attention_resunet_model.hdf5
# - final_*_model.hdf5 (×3)
# - *_history.csv (×3)
# - training_summary.csv
# - training_console.log

# 3. View summary
cat $OUTPUT/training_summary.csv

# 4. Check best performance
grep "best_val_jacard" $OUTPUT/training_summary.csv
```

### Download Results to Local Machine

```bash
# From local machine
OUTPUT_DIR="microbead_training_20251009_XXXXXX"  # Use actual timestamp

# Download entire training directory
scp -r phyzxi@hpc:~/scratch/unet-HPC/$OUTPUT_DIR ./

# Or download specific files
scp phyzxi@hpc:~/scratch/unet-HPC/$OUTPUT_DIR/training_summary.csv ./
scp phyzxi@hpc:~/scratch/unet-HPC/$OUTPUT_DIR/*_history.csv ./
scp phyzxi@hpc:~/scratch/unet-HPC/$OUTPUT_DIR/best_*.hdf5 ./
```

---

## Troubleshooting

### Issue 1: "train_microbead_optimized.py not found"

```bash
# On HPC, check if file is there
ls -la train_microbead_optimized.py

# If missing, re-upload
# From local: scp train_microbead_optimized.py phyzxi@hpc:~/scratch/unet-HPC/
```

### Issue 2: "dataset_microscope not found"

```bash
# Verify dataset location
ls -la dataset_microscope/

# If in wrong location, move it
mv /path/to/dataset_microscope ~/scratch/unet-HPC/
```

### Issue 3: Job fails immediately

```bash
# Check PBS output file
cat Microbead_Optimized_Training.o<JOBID>

# Common causes:
# - Missing models.py → copy from 224_225_226_models.py
# - Wrong paths → check all paths in script
# - Container not found → check Singularity image path
```

### Issue 4: Out of memory

```bash
# Edit train_microbead_optimized.py
# Change: BATCH_SIZE = 32
# To:     BATCH_SIZE = 16  # or even 8

# Re-submit job
```

---

## Performance Comparison

### Previous Training (Mitochondria Hyperparameters)

```
Location: microscope_training_20251008_074915/

Results:
  Best Val Jaccard: 0.1427 (Attention ResUNet, epoch 1)
  Final Val Jaccard: ~0.0 (collapsed)
  Status: ❌ FAILED
```

### New Training (Microbead-Optimized Hyperparameters)

```
Location: microbead_training_YYYYMMDD_HHMMSS/ (to be created)

Expected:
  Best Val Jaccard: 0.50-0.70 (epoch 20-50)
  Final Val Jaccard: 0.45-0.65 (stable)
  Status: ✓ SUCCESS
```

**Target:** >3.5× improvement (from 0.14 to >0.50)

---

## Quick Reference Commands

```bash
# === ON HPC ===

# Navigate to project
cd ~/scratch/unet-HPC

# Submit job
qsub pbs_microbead_optimized.sh

# Check status
qstat -u phyzxi

# Watch progress
tail -f Microbead_Optimized_Training.o<JOBID>

# After completion
ls -ltd microbead_training_*
cat microbead_training_*/training_summary.csv


# === ON LOCAL MACHINE ===

# Download results
scp -r phyzxi@hpc:~/scratch/unet-HPC/microbead_training_* ./

# Analyze locally
python analyze_training_results.py microbead_training_*/
```

---

## Summary Checklist

Before submitting job, verify:

- [x] ✅ `train_microbead_optimized.py` uploaded to HPC
- [x] ✅ `pbs_microbead_optimized.sh` uploaded to HPC
- [x] ✅ `models.py` exists on HPC
- [x] ✅ `dataset_microscope/images/` and `dataset_microscope/masks/` exist on HPC
- [x] ✅ Image/mask count matches (73 images, 73 masks)
- [x] ✅ In correct directory (`~/scratch/unet-HPC`)
- [x] ✅ Ready to submit: `qsub pbs_microbead_optimized.sh`

**Expected training time:** 6-10 hours
**Expected result:** Val Jaccard 0.50-0.70 (stable, no collapse)

---

**Good luck with the training!** 🚀

The corrected hyperparameters should fix the validation collapse issue and achieve much better performance on your dense microbead dataset.
