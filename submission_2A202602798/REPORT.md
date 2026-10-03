# Báo cáo Lab Day 1 — <Hoàng Đức Dũng> — 2A202602798

## 1. Thiết lập

- **Môi trường:** Google Colab, GPU Tesla T4, PyTorch 2.11.0+cu130. Toàn bộ chạy bằng `code/lab.ipynb` (Run all, output giữ nguyên).
- **Dữ liệu:** Forest CoverType; `train` 464 809 / `eval` 116 203 theo `split_metadata.csv` (chạy `scripts/split_data.py`, không sửa metadata). Validation: 20% của train (phân tầng, seed 42) → 371 847 train / 92 962 val. Chuẩn hoá 10 cột số bằng mean/std **chỉ của phần train còn lại** (sau chuẩn hoá: |mean| ≤ 3,5e-7, std ∈ [0,9998; 1,0008]); 44 cột nhị phân giữ nguyên. Tỉ lệ lớp ở train/val/eval trùng nhau tới 0,01%.
- **Model:** `M-base` 54→256→128→7, 47 879 tham số (có `assert`), ReLU, không softmax trong model.
- **Baseline:** CE, SGD+momentum 0,9, **lr = 0,3** (chọn bằng val, mục 2), batch 512, 20 epoch, khởi tạo He (`kaiming_normal_`, bias 0), không dropout/clip, FP32.
- **Đo mỗi epoch:** train loss (eval mode, 50 000 mẫu train cố định), val loss/acc/macro-F1, `grad_norm` toàn cục **trước clip** (trung bình và max), thời gian epoch (có `cuda.synchronize`), bộ nhớ GPU đỉnh. Metric báo cáo lấy ở epoch có val loss thấp nhất.
- **Mốc tham chiếu:** accuracy "đoán lớp đa số" trên val = **0,4876**.
- **Chủ đề đã thử:** ☑ loss ☑ optimizer ☑ hyper-parameter ☑ dropout ☑ clipping ☑ mixed precision ☑ init — 44 lần chạy, mỗi lần một dòng trong `experiments.xlsx` và một ảnh `figures/<exp_id>.png`.

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) |
| Loss bước 0 (so với ln 7 = 1,946) | 2,207 (seed 0); các seed baseline: 2,269 / 1,978 / 1,901 |
| Quá khớp 20 mẫu: loss cuối | 2,198 → **4,75e-4** sau 300 bước (Adam 1e-3), acc 100% |
| Mọi tham số có gradient khác 0 | ☑ có (‖∇W1‖ = 0,45; ‖∇W2‖ = 1,76; ‖∇W3‖ = 1,94; bias đều > 0) |
| Baseline, số seed đã chạy | 3 (`base-s1`, `base-s2`, `base-s3`) |
| Baseline: val acc (TB ± σ) | 0,9118 ± 0,0027 |
| Baseline: val macro-F1 (TB ± σ) | 0,8607 ± 0,0030 |

**Loss bước 0 cao hơn ln 7 khoảng 0,0–0,3**, vì He được áp cả cho lớp ra nên logits có std ≈ 0,6 thay vì ≈ 0, softmax ban đầu không đều. Với `zeros`/`normal` (logits ≈ 0), loss bước 0 = 1,9459/1,9460 đúng bằng ln 7, xác nhận nhãn 0..6 và hàm loss đúng.

**Ngưỡng nhiễu dùng trong báo cáo:** 2σ = **0,0060** (val macro-F1, 3 seed baseline). Ba seed là ước lượng thô; σ chỉ đo cho baseline SGD+momentum, chưa đo cho các bộ tối ưu khác.

**Chọn lr baseline** (seed 1, val macro-F1; `compare_lr_sgdm.png`): 0,01 → 0,761; 0,03 → 0,817; 0,1 → 0,839; **0,3 → 0,857**. Đường cong baseline (`base-s1.png`, `compare_baseline_seeds.png`): train và val loss cùng giảm tới epoch 20, khoảng cách val − train chỉ 0,025–0,029, best epoch 16–19. Mô hình **chưa hội tụ, chưa quá khớp**.

## 3. Kết quả theo chủ đề

Mọi so sánh trên **val**, seed 1, 20 epoch; Δ = chênh val macro-F1 so với TB baseline 0,8607. "Vượt nhiễu" nghĩa là |Δ| > 0,0060.

### 3.1 Hàm mất mát — CE vs MSE
- **Dự đoán:** gradient MSE theo logit là `2(z−y)/(7B)`, nhỏ hơn và không tập trung vào mẫu sai như CE (`softmax−y`). MSE sẽ kém rõ, nhất là ở lớp hiếm; tăng lr ×3 bù được một phần.
- **Kết quả** (`compare_loss.png`): `loss-mse` macro-F1 **0,7726** (Δ −0,088, vượt nhiễu), acc 0,8803 (−0,031). `loss-mse-lr3x` (lr 0,9): 0,7485, còn kém hơn.
- **Giải thích:** grad_norm của MSE ≈ 0,06–0,07 so với CE ≈ 0,35. Macro-F1 giảm gấp khoảng 3 lần acc, nên lớp hiếm chịu thiệt nhất: MSE kéo mọi logit về 0/1, phạt cả các lớp sai đã đủ thấp, trong khi CE dồn gradient vào lớp đúng đang có xác suất thấp (không bão hoà). Tăng lr không giúp. Khác dự đoán: lr 0,9 với momentum làm các epoch đầu dao động (F1 epoch 1 = 0,32), cho thấy vấn đề là **hướng** gradient chứ không chỉ độ lớn. Không so loss CE và MSE trực tiếp vì khác thang đo.

### 3.2 Bộ tối ưu hoá
- **Dự đoán:** ở lr tốt nhất mỗi bộ: Adam ≈ AdamW ≥ SGD+momentum > SGD; SGD thuần cần lr lớn hơn khoảng 10 lần; AdamW với wd = 0 trùng Adam.

| Bộ tối ưu (lr tốt nhất) | exp_id | lr | val macro-F1 | Δ | best epoch |
|---|---|---|---|---|---|
| Adam | `opt-adam-lr0.003` | 3e-3 | **0,8677** | +0,0070 (vượt sát ngưỡng) | 19 |
| AdamW (wd 0,01) | `opt-adamw-lr0.003` | 3e-3 | 0,8646 | +0,0039 (trong nhiễu) | 18 |
| SGD+momentum | `base-s1` (TB 3 seed) | 0,3 | 0,8573 (0,8607) | — | 19 |
| SGD | `opt-sgd-lr1` | 1,0 | 0,8344 | −0,0263 (vượt) | 18 |

- **Độ nhạy lr** (`compare_optimizer_all_lr.png`): SGD+momentum 0,761 → 0,857 khi lr 0,01 → 0,3; Adam 0,788 → 0,868 khi lr 3e-4 → 3e-3; SGD 0,755 → 0,834 khi lr 0,1 → 1,0. Không bộ nào phân kỳ trong lưới đã thử.
- **Giải thích:** Adam chia bước cho √v̂ nên mỗi tham số đi bước cỡ lr, lên nhanh hơn ở 3–5 epoch đầu (`compare_optimizer.png`). SGD lr 1,0 (0,834) ≈ SGD+momentum lr 0,1 (0,839), khớp với lr hiệu dụng `lr/(1−μ)` = 10·lr. AdamW wd = 0 cho **best_val_loss 0,249411, trùng Adam lr 1e-3 đến 6 chữ số**, tức hai bộ chỉ khác ở cách áp weight decay. Kết luận chắc chắn được: SGD thuần kém nhất. Adam hơn SGD+momentum chỉ sát ngưỡng 2σ, và Adam vs AdamW **chưa kết luận được**. Hạn chế: lr tốt nhất của Adam và SGD nằm ở đầu trên lưới.

### 3.3 Hyper-parameter
Cùng 20 epoch nhưng số bước khác nhau: batch 128 → 58 120 bước, 512 → 14 540, 2048 → 3 640 (`compare_hparam_batch.png`, `compare_hparam_arch.png`).

| exp_id | thay đổi | val macro-F1 | Δ | s/epoch |
|---|---|---|---|---|
| `hp-bs128` | batch 128, giữ lr | 0,7951 | −0,066 | 5,01 |
| `hp-bs2048` | batch 2048, giữ lr | 0,8338 | −0,027 | 0,36 |
| `hp-bs2048-lr4x` | batch 2048, lr 1,2 | 0,0936 (sụp) | −0,767 | 0,34 |
| `hp-wide` | M-wide | **0,8747** | +0,014 | 1,26 |
| `hp-deep` | M-deep | 0,8413 | −0,019 | 1,41 |
| `hp-wd5e-4` | weight decay 5e-4 | 0,7035 | −0,157 | 1,32 |

- **Batch nhỏ không tốt hơn** dù gấp 4 bước, trái với dự đoán: giữ lr 0,3 khi batch giảm 4 lần làm phương sai gradient tăng khoảng 4 lần (tỉ lệ lr/batch tăng 4 lần). Val dao động, grad_norm max có gai ~5. Batch 2048 nhanh hơn 3,5 lần mỗi epoch nhưng có 1/4 số bước nên kém hơn, đúng dự đoán.
- **Quy tắc tăng lr theo lô không warmup thất bại:** lr 1,2 tạo gai grad_norm ≈ 14 ở epoch 1. Sau đó grad ≈ 0,03 và val loss kẹt ở **1,205 = entropy của phân bố lớp**: ReLU chết, mạng chỉ còn bias và đoán phân bố tiên nghiệm. Cờ `diverged` = N vì loss vẫn hữu hạn; đây là kiểu thất bại tiêu chí NaN không bắt được.
- **M-wide hơn baseline và vượt nhiễu**, cùng thời gian/epoch (GPU chưa bão hoà), khớp chẩn đoán thiếu khớp. M-deep có F1 thấp hơn nhưng best_val_loss 0,2222 vẫn nằm trong khoảng 3 seed baseline (0,2145–0,2318); chỉ 1 seed nên **chưa kết luận được**. Weight decay 5e-4 làm train loss tăng từ 0,206 lên 0,380: với lr 0,3, mỗi bước co trọng số theo hệ số 1,5e-4, nhân qua 14 540 bước, làm mô hình vốn thiếu khớp càng thiếu khớp.

### 3.4 Dropout
- val macro-F1: q = 0,1 → 0,8385 (Δ −0,022); 0,3 → 0,7824; 0,5 → 0,5784, cả ba vượt nhiễu (`compare_dropout.png`).
- Khoảng cách val − train (train đo ở eval mode): 0,028 → 0,015 → 0,006 → 0,004, thu hẹp đúng như lý thuyết. Nhưng khoảng cách vốn đã rất nhỏ, mô hình **không quá khớp**, nên dropout chỉ làm tăng train loss (0,206 → 0,421) và giảm năng lực. Macro-F1 giảm nhanh hơn acc nhiều (q = 0,5: acc 0,820, F1 0,578) vì lớp hiếm bị ảnh hưởng trước. Khớp dự đoán về hướng; riêng q = 0,1 cũng kém rõ (dự đoán là trong nhiễu).

### 3.5 Gradient clipping
- **Ở lr bình thường:** grad_norm của base-s1 có p50 = 0,347, p90 = 0,406, p99 = 0,490, max = 2,84 (gai ở epoch 1). Chọn **c = 0,35**, clip kích hoạt ở **66,6%** số bước. Kết quả 0,8581 (Δ −0,0026, trong nhiễu): clip chỉ co bước 10–20%, gần như giảm nhẹ lr.
- **Ở lr cao** (`compare_clipping.png`): lr 3,0 (×10) không clip thì epoch 1 có gai grad_norm ≈ **7 800**, rồi sụp về đoán lớp đa số (F1 0,094). Có clip c = 0,35 thì F1 0,193, val loss 0,966 so với 1,209: đỡ hơn một chút nhưng **vẫn hỏng**. lr 9,0 (×30) thì cả hai đều hỏng.
- **Giải thích (khác dự đoán):** clipping chặn ‖g‖ ≤ c, nhưng với momentum 0,9 bước ổn định vẫn là `lr·c/(1−μ)` ≈ 3·0,35·10 ≈ 10,5, quá lớn so với độ lớn trọng số. Pre-activation bị đẩy âm, ReLU chết, gradient về ≈ 0 (nên tỉ lệ bước bị clip ở `clip-lr10x-c0.35` chỉ còn 3,3%). Clipping chữa được **gai hiếm**, không chữa được lr quá lớn một cách hệ thống.

### 3.6 Mixed precision (Tesla T4)

| exp_id | precision | model | s/epoch | peak MB | val macro-F1 |
|---|---|---|---|---|---|
| `base-s1` | FP32 | M-base | 1,255 | 28,8 | 0,8573 |
| `amp-fp16` | FP16 + GradScaler | M-base | 1,736 | 28,8 | 0,8597 |
| `amp-bf16` | BF16 | M-base | 1,545 | 28,8 | 0,8520 |
| `hp-wide` | FP32 | M-wide | 1,257 | 46,1 | 0,8747 |
| `amp-fp16-wide` | FP16 + GradScaler | M-wide | 1,725 | 46,1 | 0,8745 |

- FP16 **chậm hơn 38%**, BF16 chậm hơn 23%, kể cả với M-wide; bộ nhớ đỉnh không đổi. Độ chính xác: FP16 trong nhiễu; BF16 −0,0087 vừa vượt 2σ với 1 seed, nên chưa đủ bằng chứng là BF16 kém hơn.
- **Giải thích:** một bước chỉ vài MFLOP, nên thời gian chủ yếu là chi phí gọi kernel và Python. Autocast thêm kernel ép kiểu; GradScaler thêm unscale, kiểm tra inf và update. T4 không có phần cứng BF16 (giả lập). Bộ nhớ không giảm vì kích hoạt lô 512 chỉ khoảng 0,5 MB, còn đỉnh bị chi phối bởi bước đánh giá FP32 (lô 8 192) và tập con 50k mẫu. FP16 cần nhân loss với `s` vì chỉ có 5 bit mũ (giá trị dương nhỏ nhất ≈ 6e-8): gradient nhỏ dễ underflow về 0. BF16 có 8 bit mũ như FP32 nên thường không cần.

### 3.7 Khởi tạo tham số

| exp_id | std ReLU1 | std ReLU2 | std logits | loss bước 0 | val macro-F1 |
|---|---|---|---|---|---|
| `base-s1` (he) | 0,390 | 0,366 | 0,577 | 2,269 | 0,8573 |
| `init-zeros` | 0 | 0 | 0 | 1,9459 | 0,0936 |
| `init-normal` | 0,020 | 0,0022 | 0,0003 | 1,9460 | 0,8609 |
| `init-xavier` (2/(n_in+n_out)) | 0,163 | 0,125 | 0,192 | 2,022 | 0,8502 |
| `init-default` | 0,160 | 0,068 | 0,059 | 1,983 | 0,8637 |

- **zeros hỏng đúng như dự đoán:** mọi kích hoạt ẩn = ReLU(0) = 0, nên gradient của W1, W2, W3 bằng 0 và các nơ-ron cùng lớp luôn giống nhau (không phá được đối xứng). Chỉ bias lớp ra học, nên mạng đoán phân bố lớp: val loss 1,205, acc 0,4876.
- **normal 0,01 không kém như dự đoán** (0,8609, trong nhiễu): std giảm khoảng 10 lần mỗi lớp, nhưng mạng chỉ 3 lớp và lr 0,3 + momentum kéo trọng số về thang hợp lý ngay epoch đầu. Xavier (Δ −0,0105, 1 seed) và default (trong nhiễu) khác He ít.
- **Mạng 30 lớp** (forward ở bước 0, `compare_init_depth30.png`): std sau ReLU lớp 30: he 0,27, default 0,024, xavier 4,8e-6, normal 0. He giữ phương sai nhờ hệ số 2 bù việc ReLU bỏ nửa phân phối; Xavier (Var = 1/n_in khi n_in = n_out) mất khoảng ½ phương sai mỗi lớp. Đây là khác biệt chỉ thấy rõ khi mạng sâu, đúng như biểu đồ "30 lớp ReLU" của slide. `default` không tắt hẳn nhờ bias khác 0 của `nn.Linear`.

## 4. Đánh giá cuối trên tập eval

**Cách chọn cấu hình cuối (chỉ bằng val, định sẵn trước khi xem eval):** lấy bộ tối ưu có val macro-F1 cao nhất (Adam lr 3e-3), kết hợp với mô hình lớn hơn và 40 epoch. Ba ứng viên seed 1: `fin-wide-e40-s1` 0,9052; `fin-deep-e40-s1` 0,8946; **`fin-wide-cos-e40-s1` 0,9267** (M-wide + Adam 3e-3 + cosine lr theo bước, 40 epoch). Chọn ứng viên thứ ba, chạy thêm seed 2 và 3: val macro-F1 = **0,9264 ± 0,0010**. Nộp mô hình **seed 1** (định trước), trọng số ở epoch có val loss thấp nhất (epoch 39). Cấu hình cuối đổi nhiều yếu tố cùng lúc nên không tách được đóng góp của từng yếu tố.

| Cấu hình | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|
| Baseline (`base-s1`) | 1 | 0,8573 | **0,8565** | 0,9089 |
| Cấu hình cuối (`fin-wide-cos-e40-s1`) | 1 | 0,9267 | **0,9294** | 0,9531 |

- Cải thiện trên eval: **+0,0730** macro-F1. Trên val, chênh giữa TB 3 seed cuối và TB 3 seed baseline là +0,066, gấp khoảng 11 lần ngưỡng 2σ = 0,006, nên vượt nhiễu rõ. Điểm eval chỉ có 1 seed cho mỗi cấu hình (không chạy eval cho các seed khác).
- Val và eval rất gần nhau: baseline −0,0008, cấu hình cuối +0,0027, nên val là ước lượng đáng tin cho eval.

### 4.1 Phân tích lỗi theo lớp (cấu hình cuối, từ `eval_result.json`)

| Lớp | support | precision | recall | F1 | F1 baseline |
|---|---|---|---|---|---|
| 0 Spruce/Fir | 42 368 | 0,9524 | 0,9472 | 0,9498 | 0,9044 |
| 1 Lodgepole Pine | 56 661 | 0,9573 | 0,9632 | 0,9602 | 0,9263 |
| 2 Ponderosa Pine | 7 151 | 0,9539 | 0,9547 | 0,9543 | 0,8943 |
| 3 Cottonwood/Willow | 549 | 0,8969 | 0,8561 | **0,8760** | 0,8015 |
| 4 Aspen | 1 899 | 0,9086 | 0,8794 | 0,8938 | 0,7595 |
| 5 Douglas-fir | 3 473 | 0,9200 | 0,9110 | 0,9155 | 0,8089 |
| 6 Krummholz | 4 102 | 0,9570 | 0,9559 | 0,9565 | 0,9002 |

Ma trận nhầm lẫn (hàng = thật, cột = dự đoán):

```
       0      1     2    3     4     5     6
0  40129   2054     1    0    24     2   158
1   1824  54576    61    0   125    57    18
2      2     95  6827   32    13   182     0
3      0      0    54  470     0    25     0
4     31    180     9    0  1670     9     0
5      2     75   205   22     5  3164     0
6    147     33     0    0     1     0  3921
```

- **Lớp khó nhất là lớp 3 Cottonwood/Willow (F1 = 0,876).** Nó hay bị nhầm với **lớp 2 Ponderosa Pine (54/549 = 9,8%)** và lớp 5 Douglas-fir (25 mẫu = 4,6%). Lớp khó thứ hai là lớp 4 Aspen (F1 = 0,894), bị nhầm sang lớp 1 Lodgepole Pine 9,5% (180 mẫu). Cặp nhầm lớn nhất về số lượng là 0 ↔ 1 (2 054 + 1 824 mẫu).
- **Lý giải:**
  - *Đã đo:* lớp 3 chỉ có 1 759 mẫu train (0,47%), lớp 4 có 6 075 (1,6%). Với CE trung bình, gradient bị lớp 0 và 1 (85% dữ liệu) chi phối. Cũng vì vậy hai lớp này có F1 thấp nhất ở cả baseline lẫn cấu hình cuối.
  - *Từ mô tả bộ dữ liệu (Blackard & Dean, 1999), mình chưa tự kiểm chứng trên đặc trưng:* Cottonwood/Willow, Ponderosa Pine và Douglas-fir là các loài ở độ cao thấp, cùng xuất hiện ở khu Cache la Poudre, nên đặc trưng địa hình chồng lấn.
- **Cách cải thiện sẽ thử:** CE có trọng số lớp (∝ 1/√tần suất) hoặc lấy mẫu cân bằng cho lớp 3, 4. Cấu hình đó chọn bằng val macro-F1, không bằng eval.

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào "thắng" khi chỉnh lr công bằng?** Adam (lr 3e-3, 0,8677) cao nhất, nhưng chỉ hơn TB baseline SGD+momentum 0,007, sát ngưỡng 2σ = 0,006. AdamW (0,8646) không phân biệt được với Adam. Chỉ có thể nói chắc rằng SGD thuần kém nhất (−0,026). Khi không chỉnh lr, kết luận đảo chiều tuỳ cặp lr: Adam 1e-3 "thắng" SGD+momentum 0,01 tới 0,084, nhưng Adam 3e-4 "thua" SGD+momentum 0,3 tới 0,069. So sánh ở một lr duy nhất thực chất là so lr, không phải so bộ tối ưu.
2. **Dropout có giúp khi mô hình chưa quá khớp không?** Không. Ở đây khoảng cách val − train chỉ 0,028, và dropout 0,1 / 0,3 / 0,5 đều làm macro-F1 giảm (−0,022 đến −0,282) vì cắt bớt năng lực của một mô hình vốn đang thiếu khớp. Nên dùng dropout khi train loss tiếp tục giảm còn val loss tăng (khoảng cách lớn dần), và chọn q bằng val.
3. **Gradient clipping giải quyết vấn đề gì? Quan sát nào chứng minh?** Clipping chặn các **gai gradient hiếm**: base-s1 có max 2,84 so với p99 0,49, nên clip c = 0,35 giữ các bước ở epoch 1 trong giới hạn mà không đổi kết quả (Δ −0,003, trong nhiễu). Thí nghiệm lr ×10 cho thấy giới hạn của nó: gai ≈ 7 800 bị chặn nhưng mạng vẫn sụp (F1 0,19), vì momentum vẫn tích luỹ bước `lr·c/(1−μ)` ≈ 10. Clipping không thay được việc chọn lr đúng hoặc warmup.
4. **Mixed precision có nhanh hơn trên mạng và dữ liệu này không?** Không. Trên T4, FP16 chậm hơn 38%, BF16 chậm hơn 23%, kể cả với M-wide, và bộ nhớ đỉnh không giảm. Mạng 48k–161k tham số với lô 512 chỉ vài MFLOP mỗi bước, nên thời gian chủ yếu là chi phí gọi kernel; autocast và GradScaler còn thêm kernel. Mixed precision có lợi khi phép nhân ma trận đủ lớn để Tensor Core là nút thắt (mạng hoặc lô lớn hơn nhiều).
5. **Vì sao khởi tạo toàn 0 hỏng? He khác Xavier thế nào?** Với W = 0, mọi kích hoạt ẩn bằng ReLU(0) = 0. Gradient của mọi W bằng 0 và các nơ-ron trong một lớp luôn giống hệt nhau, nên mạng chỉ học được bias lớp ra, tức phân bố lớp (loss kẹt ở 1,205, acc 0,4876). He dùng Var = 2/n_in để bù việc ReLU bỏ một nửa phương sai; Xavier dùng Var ≈ 1/n_in, nên qua mỗi lớp ReLU mất khoảng ½ phương sai. Ở mạng 3 lớp khác biệt nhỏ (trong hoặc sát nhiễu); ở mạng 30 lớp, std lớp cuối là 0,27 (He) so với 4,8e-6 (Xavier). Điều này quan trọng khi mạng sâu.
6. **Loss không giảm sau 2 000 bước: 3 phép kiểm tra đầu tiên.**
   1. **So loss hiện tại với ln C và với entropy của phân bố lớp.** Loss bước 0 phải ≈ ln 7 = 1,946; lệch nhiều là lỗi khởi tạo, chuẩn hoá hoặc nhãn. Nếu loss **kẹt đúng ở entropy phân bố lớp (≈ 1,205 ở bài này)** thì mạng đã sụp về chỉ học bias, như `init-zeros`, `hp-bs2048-lr4x`, `clip-lr10x-noclip`: cả ba kẹt ở 1,205 và acc 0,4876. Lỗi khi đó nằm ở khởi tạo hoặc lr, không phải dữ liệu.
   2. **Quá khớp một lô nhỏ (20 mẫu, tắt chính quy hoá).** Nếu loss không về ≈ 0 thì gần như chắc chắn là lỗi vòng lặp huấn luyện (nhãn lệch, softmax hai lần, quên `zero_grad`, tham số không nằm trong optimizer). Pipeline của bài này qua được phép thử (4,75e-4), nên lỗi khi đó không nằm ở code.
   3. **In `grad_norm` theo lớp và theo thời gian, rồi quét lr 3–10 lần.** grad ≈ 0 ở mọi lớp sau một gai lớn là dấu hiệu lr quá cao làm ReLU chết (bài này: gai 7 800 → grad 0,03). Gradient đều nhỏ và loss giảm rất chậm là lr quá thấp (SGD+momentum lr 0,01 chỉ đạt 0,761). Nhìn gai và độ lớn gradient cho biết nên giảm lr, thêm warmup hay đổi khởi tạo.

## 6. Hạn chế và điều bất ngờ

- **Khác dự đoán:**
  - Batch 128 kém hơn batch 512 (do giữ lr; đúng ra phải giảm lr).
  - Quy tắc tăng lr ×4 theo lô làm mạng sụp (thiếu warmup).
  - Clipping không cứu được lr ×10 (momentum khuếch đại bước đã bị clip).
  - Init `normal` không kém He ở mạng 3 lớp.
  - M-deep có F1 thấp hơn baseline.
  - MSE + lr ×3 còn tệ hơn MSE.
  - Weight decay 5e-4 gây hại mạnh hơn nhiều so với dự đoán.
- **Có thể làm sai kết luận:**
  - Mỗi thí nghiệm chỉ 1 seed. σ chỉ đo trên 3 seed của baseline SGD+momentum, chưa đo cho Adam hay các mô hình khác.
  - Cùng số epoch nhưng số bước khác nhau ở thí nghiệm batch.
  - lr tốt nhất của SGD+momentum (0,3), Adam (3e-3) và SGD (1,0) đều nằm ở đầu lưới, nên chưa chắc là tối ưu.
  - Các thí nghiệm ở chủ đề khác dùng lr baseline 0,3 khá lớn, nên vài kết quả (M-deep, xavier) có thể phụ thuộc lr.
  - Cấu hình cuối thay nhiều yếu tố cùng lúc.
  - Cờ `diverged` chỉ bắt NaN/inf, không bắt được trường hợp mạng sụp về đoán lớp đa số; các lần sụp được ghi trong `notes` và trong báo cáo.
  - Peak memory tính cả tập con 50k mẫu tạo trong `run_experiment`.
- **Nếu có thêm thời gian:**
  - Chạy 3 seed cho Adam và M-wide để có σ riêng.
  - Mở rộng lưới lr và chỉnh lr theo batch (128 → 0,075).
  - Thêm warmup cho batch lớn.
  - CE có trọng số lớp cho lớp 3, 4.
  - Thử M-wide + cosine với 60–80 epoch (val vẫn giảm ở epoch 40).

## 7. Phụ lục

- **File đã nộp:** `REPORT.md`, `experiments.xlsx` (44 dòng; sheet Seeds = base-s1..s3; Summary có nhận xét theo nhóm), `predictions_eval.csv` (116 203 dòng), `eval_result.json`, `figures/` (44 ảnh `<exp_id>.png` + 13 ảnh `compare_*.png`), `results/` (44 file `<exp_id>.json`, cùng `predictions_eval_base-s1.csv` và `eval_result_base-s1.json` của baseline), `code/` (`lab.ipynb` có output, `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`).
- **Thời gian chạy:** tổng thời gian huấn luyện (Σ time_per_epoch × epochs) ≈ 25 phút trên T4. Chưa tính thời gian đánh giá val/train mỗi epoch, nên toàn bộ notebook mất lâu hơn con số này.
