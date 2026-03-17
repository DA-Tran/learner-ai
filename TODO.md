# Model Reordering Plan (rnn → lstm → gan → lgbm → xgb)

**✅ Plan Approved** by user.

**Files to Update:**
1. **Main.py** `train_single_dataset()`:
   - CV section first (rnn/lstm only ✓)
   - **NEW ORDER**: "\nNEURAL MODELS" (rnn → lstm → gan)
   - Then "\nENSEMBLE MODELS" (lgbm → xgb)
   - RESULTS print: "rnn | lstm | gan | lgbm | xgb"

2. **Main.py** `main()` interactive:
   - Print: "Available: rnn, lstm, gan, lgbm, xgb"
   - 'all': `models = ['rnn', 'lstm', 'gan', 'lgbm', 'xgb']`

3. **plot_utils.py** `plot_single_dataset_comparison()`:
   - `models = ['RNN', 'LSTM', 'GAN', 'LGBM', 'XGB']`
   - Colors/indices match new order

4. **Minor**: Update plot titles/comments "5 Models: RNN/LSTM/GAN/LGBM/XGB"

**Followup Steps:**
1. Apply edits
2. Test `./run_clean.sh 1 iris rnn,lstm` → check order/prints
3. Test 'all' → verify plots (bars left→right: RNN/LSTM/GAN/LGBM/XGB)
4. `rm iris_comparison.png` between tests → clean

**Ready for edits** → test → complete!
