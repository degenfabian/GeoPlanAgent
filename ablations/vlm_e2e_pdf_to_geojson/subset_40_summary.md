# VLM-E2E subset (N=40)

- source: `evaluation_data/new_updated.xlsx` → `Cleaned_up_208_planning_dataset`
- filter: `Shape Matches correctly` ∈ {`yes`, `yes - across …`}
- stratification: Document Quality × Shape Complexity, floor=2
- seed: 42

## Stratum allocation

| Stratum | Pop | Allocated | Pipeline mean IoU |
|---|---|---|---|
| bad_x_easy | n/a | 4 | 0.576 |
| bad_x_hard | n/a | 3 | 0.420 |
| bad_x_medium | n/a | 3 | 0.899 |
| good_x_easy | n/a | 15 | 0.672 |
| good_x_hard | n/a | 3 | 0.758 |
| good_x_medium | n/a | 12 | 0.852 |

**Overall pipeline mean IoU on the 40 subset cases: 0.721**

## Cases (alphabetical)

- `095AB379-F04E-473A-BC0D-8948B58E4090` — bad_x_easy
- `115` — good_x_medium
- `118` — good_x_easy
- `12:00115:ART4` — good_x_easy
- `12:00125:ART4` — bad_x_hard
- `12:00140:ART4` — good_x_medium
- `12:00154:ART4` — good_x_easy
- `12:00156:ART4` — good_x_medium
- `12:00161:ART4` — bad_x_hard
- `12:00162:ART4` — good_x_medium
- `22` — bad_x_easy
- `23:53161:ART4` — good_x_medium
- `3DA282A7-E829-47CF-B842-E03E0C704072` — good_x_hard
- `498E1484-1D2E-418C-9547-A93AE9A57BB0` — good_x_easy
- `4AB36890-E52B-4CCC-9CDE-FB1476FCEB82` — bad_x_easy
- `5B10B5A8-B0A0-4DB3-9867-55B78F678079` — good_x_easy
- `69` — good_x_easy
- `A4D-04` — bad_x_medium
- `A4D-21` — good_x_medium
- `A4D-24` — good_x_medium
- `A4DA01` — good_x_medium
- `A4Ha1` — good_x_hard
- `A4Sa1` — good_x_hard
- `A4_088:LL:016` — good_x_medium
- `ART4:66:00001` — good_x_easy
- `Ar4.15` — good_x_medium
- `Ar4.2` — good_x_easy
- `Ar4.25` — good_x_easy
- `Ar4.27` — good_x_easy
- `Ar4.5` — bad_x_medium
- `Ar4.8` — bad_x_easy
- `Art4D05` — good_x_medium
- `B6C8BCAD-105E-4CA8-83B3-5A20DA5602EE` — bad_x_medium
- `CPA4(4a)` — good_x_easy
- `DE5A30DA-29A4-45BE-B60A-C201A5F11C6F` — good_x_easy
- `REG_4:36880` — bad_x_hard
- `SSA401` — good_x_easy
- `SSA410` — good_x_easy
- `SSA414` — good_x_medium
- `SSA415` — good_x_easy