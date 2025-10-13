# Architecture Comparison - Quick Start Card

## 🚀 Submit to HPC (Fast Path)

```bash
# 1. Transfer files to HPC
scp pbs_architecture_comparison.sh validate_architecture_comparison.py \
    model_architectures.py loss_functions_fixed.py \
    phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/

# 2. SSH and submit
ssh phyzxi@nscc
cd /home/svu/phyzxi/scratch/unet-HPC
qsub pbs_architecture_comparison.sh

# 3. Monitor (replace <JOB_ID> with actual ID from qsub)
qstat -u phyzxi
tail -f Architecture_Comparison.o<JOB_ID>

# 4. Download results after ~3 hours
# (on local machine)
scp -r phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/validation_arch_comparison_*/ ./

# 5. Analyze
conda activate unetCNN
python analyze_architecture_comparison.py validation_arch_comparison_YYYYMMDD_HHMMSS/
```

---

## 📊 Run Locally (Alternative)

```bash
# Activate environment
conda activate unetCNN

# Run comparison (~3 hours)
python validate_architecture_comparison.py

# Analyze results
python analyze_architecture_comparison.py validation_arch_comparison_YYYYMMDD_HHMMSS/

# View report
open validation_arch_comparison_YYYYMMDD_HHMMSS/ARCHITECTURE_COMPARISON_REPORT.md
```

---

## 📁 What You'll Get

```
validation_arch_comparison_20251013_143022/
├── architecture_comparison_summary.json          # Key results
├── ARCHITECTURE_COMPARISON_REPORT.md             # Full analysis (after running analyze script)
├── architecture_performance_comparison.png       # Plots (after analysis)
├── architecture_training_curves.png
├── architecture_convergence_analysis.png
└── [unet|resunet|attention_resunet]/
    └── fold_[1-5]/
        ├── best_model.keras      # Trained model
        ├── history.csv            # Training metrics
        └── results.json           # Fold summary
```

---

## 🎯 Key Questions Answered

1. **Which architecture performs best?**
   → Check REPORT.md "Best Performing Architecture"

2. **Is the improvement statistically significant?**
   → Check REPORT.md "Statistical Analysis" section (p-values)

3. **What's the computational cost?**
   → Check REPORT.md "Performance Comparison" table (Avg Epoch Time)

4. **Should I switch from U-Net?**
   → Check REPORT.md "Recommendations" section

---

## ⏱️ Expected Timelines

| Method | Setup | Execution | Analysis | Total |
|--------|-------|-----------|----------|-------|
| **HPC** | 5 min | 3 hours | 5 min | ~3h 10min |
| **Local** | 0 min | 3 hours | 5 min | ~3h 5min |

---

## 📖 Detailed Guides

- **HPC Submission:** `HPC_SUBMISSION_GUIDE.md` (11 KB, comprehensive)
- **Architecture Details:** `ARCHITECTURE_COMPARISON_GUIDE.md` (15 KB, educational)
- **Troubleshooting:** Both guides above

---

## 🆘 Quick Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| Job stuck in queue | Wait or submit off-peak |
| "Dataset not found" | Check `dataset_full_stack/` exists on HPC |
| OOM error | Reduce `batch_size` to 2 in script |
| Job too slow | Reduce `n_folds` to 3 in script |
| Import errors | Ensure all .py files transferred |

---

## 💡 Pro Tips

1. **Start with 3-fold CV** for faster initial results (edit script: `n_folds: 3`)
2. **Run overnight** if executing locally (takes 3 hours)
3. **Submit during off-peak hours** on HPC (evenings/weekends)
4. **Save the summary.json** even if models are large
5. **Compare to baseline:** Your U-Net CV mean is **60.97% ± 11.5%**

---

## 📞 Support

- **HPC Issues:** Check `HPC_SUBMISSION_GUIDE.md` or NSCC helpdesk
- **Code Questions:** Check `ARCHITECTURE_COMPARISON_GUIDE.md`
- **Script Errors:** Read console log and traceback

---

**Ready to find the best architecture? Submit the job! 🚀**
