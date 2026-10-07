# BHN — KPI baseline & cách đo lại

Mục đích: biết các thay đổi cấu hình/nội dung có thực sự cải thiện server hay không. Mỗi lần đo ghi thêm một dòng vào bảng dưới.

## Cách đo

```bash
cd ~/Work/FOSS/mcp-discord
.venv/bin/python docs/ops/measure_bhn_metrics.py
```

Script đọc `DISCORD_TOKEN` từ `.env`, ghi `docs/ops/bhn-metrics-<ngày>.json` và in tóm tắt. So sánh với file baseline bằng `jq`/diff.

## Baseline

| Ngày | Member (human) | MAU 30d | WAU 7d | Tin nhắn 30d | Top-3 share | Activation ≤30d | Kênh 0 tin/30d | Sidebar (member/mod) | Nguồn |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-07 | 274 | 11% (30) | 4.7% (13) | ≥699 (~23/ngày) | 42% | 38% | 26/41 | 27 / 34 | `bhn-metrics-2026-10-07.json` |
| 2026-10-08 | 274 | 11% (30) | 5% (13) | 699 (~23/ngày) | 42% | 38% | 26/42 | 27 / 34 | `bhn-metrics-2026-10-08.json` |

Hai lần đo liên tiếp cho cùng kết quả (chênh 1 kênh do danh mục đổi) → script tái lập được, dùng làm mốc so sánh.

Chi tiết 2026-10-07 (đo bằng script quét API):

- Tăng trưởng theo tháng join: 25-10 **53** · 25-12 44 · 26-04 30 · 26-07 14 · 26-08 11 · 26-09 9 · 26-10 3 → chậm dần.
- Invite: 543 lượt dùng (DISBOARD 370 = 68%) cho 274 người → churn ~50% (cận trên).
- Tập trung nội dung: top-1 16% · top-3 42% · top-10 81% — 3 người đăng nhiều nhất đều là staff.
- Kênh hoạt động: `💬 trà-đá-vỉa-hè` ≥72% tin nhắn; sau đó `❔ hỏi-mỗi-ngày` (bot QOTD) 148, `👋 chào-mừng` 49, `📸 khoảnh-khắc` 41, `✨ hành-trình` 33.
- Giờ cao điểm (giờ VN): 19–23h = 43% hoạt động; 06–11h = 24%.
- Forum: 54 thread tổng, **0 thread trong 90 ngày**.
- Moderation 30d: kick 7 (Sapphire chặn account mới), message_delete 21, automod_block 1, ban/timeout 0 → tải mod thấp, nút thắt là im lặng chứ không phải toxic.
- Level ladder: chỉ 45/274 (16%) đạt level 5; 1 người đạt level 60.

## Mục tiêu

| Mốc | MAU | WAU | Activation người mới | Tin nhắn/tuần | Top-10 share |
|---|---|---|---|---|---|
| 30 ngày | ≥15% | ≥6% | ≥45% | ≥250 | ≤75% |
| 60 ngày | ≥20% | ≥8% | ≥50% | ≥350 | ≤70% |
| 90 ngày | ≥20% | ≥10% | ≥55% | ≥450 | ≤65% |

## Quyết định đã chốt (không làm lại)

- **Không mở lại** các kênh vòng lặp đang nằm trong `📦 Kỉ niệm` (`🎡 sự-kiện`, `🎁 lộc-lá-hiên-nhà`, `🎡 nhiệm-vụ-daily`, `🎣 câu-cá`, `🎰 xì-dách`, `🌲 trồng-cây`, `🦀 bầu-cua-tôm-cá`, `🔡 nối-từ*`, `🦄 pokemeow`…). Category `📦 Kỉ niệm` deny `read_messages` cho `@everyone` → member không thấy; giữ nguyên.
- Không xoá kênh nào; các kênh chết được **gom vào `📦 Kỉ niệm`** (đã làm 2026-10-07/08) → sidebar member còn 27 kênh.
- Không gửi bài "BẮT ĐẦU TẠI ĐÂY" (task B1 đã bỏ).
- Việc còn lại thuộc dashboard: Sapphire welcome message + OpenQOTD ping người mới trong 24h.
