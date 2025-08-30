import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Bilingualdataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len)-> None:
        super().__init__()
        self.seq_len=seq_len
        self.ds=ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt
        self.lang_src=lang_src
        self.lang_tgt=lang_tgt
        # Token speciali
        self.sos_token=torch.tensor([tokenizer_src.token_to_id('[SOS]')]).to(torch.int64)
        self.eos_token=torch.tensor([tokenizer_src.token_to_id('[EOS]')]).to(torch.int64)
        self.pad_token=torch.tensor([tokenizer_src.token_to_id('[PAD]')]).to(torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair=self.ds[index]
        src_text=src_target_pair['translation'][self.lang_src]
        tgt_text=src_target_pair['translation'][self.lang_tgt]
        enc_input_tokens= self.tokenizer_src.encode(src_text).ids
        dec_input_tokens= self.tokenizer_tgt.encode(tgt_text).ids
        enc_padding_num= self.seq_len-len(enc_input_tokens)-2
        dec_padding_num= self.seq_len-len(dec_input_tokens)-1
        
        if enc_padding_num<0 or dec_padding_num<0:
            raise ValueError('Sentence is too long.')
        
        # Encoder input
        encoder_input=torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens).to(torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_padding_num).to(torch.int64)
            ]
        )

        decoder_input=torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens).to(torch.int64),
                torch.tensor([self.pad_token]*dec_padding_num).to(torch.int64)
            ]
        )

        label=torch.cat(
            [
                torch.tensor(dec_input_tokens).to(torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_padding_num).to(torch.int64)
            ]
        )

        assert encoder_input.size(0)==self.seq_len 
        assert decoder_input.size(0)==self.seq_len
        assert label.size(0)==self.seq_len

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }
    

def causal_mask(size):
    mask=torch.triu(torch.ones(1,size,size), diagonal=1).type(torch.int)
    return mask==0