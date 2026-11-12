# Model comparison (synthetic test words, EMNIST test characters)

3000 test words, 41.2 % of them with touching letters.

| Model | Word accuracy | Word acc. (ignoring case) | CER | Word acc. touching | Word acc. separated |
|---|---|---|---|---|---|
| A: segmentation + CharCNN | 16.2 % | 25.5 % | 49.1 % | 4.8 % | 24.3 % |
| B: CRNN + CTC (greedy) | 69.8 % | 74.2 % | 6.1 % | 65.2 % | 73.1 % |
| B: CRNN + CTC (beam search) | 70.1 % | 74.5 % | 6.0 % | 65.6 % | 73.2 % |

Model A found the right NUMBER of characters in 55.1 % of words (touching: 13.9 %, separated: 83.9 %).

# CharCNN on single EMNIST characters (116,323 test images)

| Top-1 | Top-3 | Balanced acc. | Ignoring case |
|---|---|---|---|
| 86.0 % | 99.1 % | 71.0 % | 90.2 % |

Most frequent confusions (true -> predicted):

| true | predicted | count | share of true class |
|---|---|---|---|
| O | 0 | 2577 | 62.0 % |
| l | 1 | 2175 | 85.8 % |
| I | 1 | 1388 | 67.8 % |
| m | M | 461 | 99.4 % |
| u | U | 449 | 93.2 % |
| c | C | 424 | 98.1 % |
| 0 | O | 409 | 7.1 % |
| v | V | 408 | 87.2 % |
| s | S | 382 | 87.4 % |
| f | F | 377 | 94.2 % |
