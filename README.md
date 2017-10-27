https://www.youtube.com/watch?v=HMq7rTo7C6M


## TODO
* data cleaning scripts
    * tokenizing, cleaning, standardizing
    * splitting, shuffling
    * vocab induction
* hyperparam search
* better word pulling from attn (**kelly**)
* better evaluations?
    * understand/use Precision in Estimation of Heterogeneous Effect (PEHE) from https://arxiv.org/pdf/1605.03661.pdf
* new paper baseline(s)?
    * understand and implement https://arxiv.org/pdf/1605.03661.pdf
    * understand and implement https://scholar.princeton.edu/sites/default/files/bstewart/files/ais.pdf
    * understand and implement https://arxiv.org/pdf/1606.03976.pdf
         * https://github.com/clinicalml/cfrnet
* think about backproping through cramer's v?
* plotting (**kelly**)
* post-training summaries
* progress bars, restarting experiments, etc etc
* impose max length on input seqs (**kelly**)
   * done for neural only
* figure out why mixed regression takes 4ever to train
* no lstm baseline?
* **baseline:** f(confounds) => Y, take residual, learn g(text) => residual

