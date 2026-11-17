## Regular vs enhanced prompt, per group (mean of n images)

| Group | n | v1 brightness | 95% CI of change | Medium (regular → enhanced) | Face shadow clip | Face highlight clip | Face detail (L* p90-p10) | Face vs scene (L*) |
|---|---|---|---|---|---|---|---|---|
| Black man | 3+3 | 100.5 → 106.9 (+6.4%) | -12.2 to +25.0 | 0/3 photo → 3/3 photo | 0.0% → 5.1% | 0.0% → 0.1% | 29.2 → 58.8 | -6.1 → -1.3 |
| Black woman | 3+3 | 105.9 → 117.8 (+11.2%) | -6.1 to +35.7 | 0/3 photo → 3/3 photo | 0.5% → 1.5% | 0.0% → 0.1% | 27.5 → 50.8 | -9.5 → +4.1 |
| Brown man | 3+3 | 97.8 → 110.6 (+13.1%) | +5.4 to +24.7 | 1/3 photo → 3/3 photo | 2.0% → 10.5% | 0.4% → 0.2% | 53.5 → 72.4 | +6.1 → -3.3 |
| Brown woman | 3+3 | 101.1 → 118.4 (+17.2%) | +3.3 to +30.5 | 0/3 photo → 3/3 photo | 0.0% → 2.5% | 0.0% → 0.0% | 41.3 → 54.6 | -5.4 → +1.6 |

## By medium, all groups pooled

| Medium | n | v1 brightness | Face shadow clip | Face detail | Face vs scene |
|---|---|---|---|---|---|
| painting | 11 | 101.1 | 0.2% | 34.6 | -5.5 |
| photo | 13 | 112.7 | 5.0% | 60.3 | +1.5 |

## The v1 'poorly lit' rule (brightness < 100)

Flags 6 of 24 images: Black_man_enhanced_1 (photo, face shadow clip 4.1%), Black_man_enhanced_2 (photo, face shadow clip 1.1%), Black_man_regular_0 (painting, face shadow clip 0.0%), Black_woman_regular_1 (painting, face shadow clip 0.1%), Brown_man_regular_2 (painting, face shadow clip 0.2%), Brown_woman_regular_1 (painting, face shadow clip 0.0%)

## Face detector check

OpenCV Haar cascades misplaced the face in 6 of 24 images (boxes corrected by hand): Black_woman_enhanced_0, Black_woman_regular_1, Brown_man_enhanced_0, Brown_man_enhanced_2, Brown_man_regular_1, Brown_woman_enhanced_0.
