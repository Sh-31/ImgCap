from torcheval.metrics.functional.text import bleu_score
from pycocoevalcap.cider.cider import Cider

def eval_bleu_score(candidates, references, n_gram=4):
  '''
   args:
      candidates: list of strings (Generated caption)
      references: list of lists of strings (Ground truth)
      n_gram: int

    inputs example:
      candidates = ["this is a test", "this is another test"]
      references = [["this is a test"], ["this is another test"]] or ["this is a test", "this is another test"]

    returns:
      bleu score
  '''

  return bleu_score(input=candidates, target=references, n_gram=n_gram).item()


def eval_CIDEr(candidates, references, list_to_dict=True, inner_list=False):
  '''
   args:
      candidates: list of strings (Generated caption)
      references: list of lists of strings (Ground truth)
      list_to_dict: bool
      inner_list: bool

    inputs example:
      candidates = ["this is a test", "this is another test"]
      references = [["this is a test"], ["this is another test"]]

      or

      candidates = {0: "this is a test", 1: "this is another test"}
      references = {0: "this is a test", 1: "this is another test"}
      list_to_dict = False
      inner_list = True

      or

      candidates = {0: ["this is a test"], 1: ["this is another test"]}
      references = {0: ["this is a test"], 1: ["this is another test"]}
      list_to_dict = False
      inner_list = False

   return:
      avg_score: float
      scores: list of floats
  '''
  if list_to_dict:
    references = {i: [references[i]] for i in range(len(references))}
    candidates = {i: [candidates[i]] for i in range(len(candidates))}

  if inner_list:
    references = { k:[v] for k,v in references.items() }
    candidates = { k:[v] for k,v in candidates.items() }

  matric = Cider()
  avg_score , scores = matric.compute_score(res=references, gts=candidates)

  return avg_score, scores

def clean_caption(caption, vocab_decoder, vocab):
    words = []
    for token in caption:
        word = vocab_decoder([token])
        if word == '<eos>':
            words.append(token)
            break
        if word not in ['<pad>']:
            words.append(token)

    return vocab_decoder(words)

def eval_decode_batch(captions, vocab_decoder, vocab):
    deconed_captions = []

    for i in range(captions.shape[0]):

        ref_caption = clean_caption(captions[i, :].tolist(), vocab_decoder, vocab)
        deconed_captions.append(ref_caption)

    return deconed_captions
