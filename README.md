# DNACSE

DNACSE: Enhancing Genomic LLMs with Contrastive Learning for DNA Barcode Identification.


## Abstract
DNA barcoding is a powerful tool for exploring biodiversity, and DNA language models have significantly facilitated its construction and identification. However, since DNA barcodes come from a specific region of mitochondrial DNA and there are structural differences between DNA barcodes and reference genomes used to train existing DNA language models, it is difficult to directly apply the existing DNA language models to the DNA barcoding task. To address this, we introduce DNACSE (DNA Contrastive Learning for Sequence Embeddings), an unsupervised noise contrastive learning framework designed to fine-tune the DNA language foundation model while enhancing the distribution of the embedding space. The results demonstrate that DNACSE outperforms direct usage of DNA language models in DNA barcoding-related tasks. Specifically, in fine-tuning and linear probe tasks, it achieves accuracy rates of 99.17\% and 98.31\%, respectively, surpassing the current state-of-the-art BarcodeBERT by 6.44\% and 6.44\%. In zero-shot clustering tasks, it raises the Adjusted Mutual Information (AMI) score to 92.25\%, an improvement of 8.36\%. In addition, zero-shot benchmarking and genomic benchmarking tests are evaluated, indicating that DNACSE enhances the performance of DNA language models in generalized genomic tasks. In summary, DNACSE has demonstrated excellent performance in DNA barcode species classification by making full use of multi-species information and DNA barcode information, providing a feasible way to further explore and protect biodiversity.

## Requirement
To train our model, go to the root directory and execute the following code to install the necessary packages.
```
pip install -r requirements.txt
```
We use Flash Attention 2 to accelerate the training and inference processes. Please execute the following command to install Flash Attention 2. For more detailed information, please refer to https://github.com/Dao-AILab/flash-attention

To install 
```
pip install flash-attn --no-build-isolation
```







## train

You can obtain our training dataset and scripts for processing the dataset from this zenodo repository:https://zenodo.org/records/16902079

```sh
bash run_unsup_example.sh
```



## Citation

If this work is helpful, please cite as:

```bibtex



```


## License

MIT
