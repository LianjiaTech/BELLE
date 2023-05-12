''' Train tokenizer'''
import sentencepiece as spm

spm.SentencePieceTrainer.train(input='/path/to/input_text', 
							   model_prefix='belle', 
							   model_type='bpe', 
							   vocab_size=25000, 
							   character_coverage=0.9995)

''' Merge tokenizer '''
import sentencepiece_model_pb2 as model
orig_model_path = '/path/to/llama/tokenizer.model'
belle_model_path = '/path/to/belle/belle.model'
orig_m = model.ModelProto()
belle_m = model.ModelProto()
orig_m.ParseFromString(open(orig_model_path, "rb").read())
belle_m.ParseFromString(open(belle_model_path, "rb").read())
print(len(orig_m.pieces), len(belle_m.pieces))
orig_pieces = []
for piece in orig_m.pieces:
    orig_pieces.append(piece.piece)
for piece in belle_m.pieces:
    if piece.piece not in orig_pieces:
        orig_m.pieces.append(piece)
        orig_pieces.append(piece.piece)

print(len(orig_m.pieces))
save_vocab_path = '/path/to/merge_tokenizer/tokenizer.model'
with open(save_vocab_path, 'wb') as f:
    f.write(orig_m.SerializeToString())