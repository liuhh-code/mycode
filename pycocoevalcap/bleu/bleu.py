
from .bleu_scorer import BleuScorer


class Bleu:
    def __init__(self, n=4):
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res, score_option = 'closest', verbose = 1):
        '''
        Inputs:
            gts - ground truths
            res - predictions
            score_option - {shortest, closest, average}
            verbose - 1 or 0
        Outputs:
            Blue scores
        '''
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)

            bleu_scorer += (hypo[0], ref)

        score, scores = bleu_scorer.compute_score(option = score_option, verbose =verbose)

        return score, scores

    def method(self):
        return "Bleu"
