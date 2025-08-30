import torch
import torch.nn as nn
import math


#Embedding vettoriale dei tokens
class InputEmbedding(nn.Module):
    def __init__(self,d_model: int, vocab_size: int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)
        
    def forward(self,x):
        return math.sqrt(self.d_model)*self.embedding(x)

#Vettori posizione
class PositionalEncoding(nn.Module):
    def __init__(self,d_model: int,seq_len: int, dropout: float):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)
        
        #Costruiamo la matrice per i vettori posizione (seq_len,d_model)
        pe=torch.zeros(seq_len,d_model)
        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        #Popoliamo la matrice pe. Si usa sin per le colonne pari e cos per quelle dispari
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0) # tensore di forma (1,seq_len,d_model)
        #self.pe=pe
        self.register_buffer('pe', pe)
        
    def forward(self,x):
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)


#Strati per il modulo Encoder
#Normalizzazione
class LayerNorm(nn.Module):
    def __init__(self,eps: float=10**-6):
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))
        self.bias=nn.Parameter(torch.zeros(1))
        
    def forward(self,x):
        mean=x.mean(dim=-1, keepdim=True)
        std=x.std(dim=-1, keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps) + self.bias

# Blocco FeedForward
class FeedForwardBlock(nn.Module):
    def __init__(self,d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout()
        self.linear2=nn.Linear(d_ff,d_model)
        
    def forward(self,x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

#Multi-head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model: int,h: int,dropout: float):
        super().__init__()
        self.d_model=d_model
        self.h=h
        self.d_k=d_model//h
        assert d_model%h==0, 'd_model is not divisible by h'
        
        self.dropout=nn.Dropout(dropout)
        self.w_q=nn.Linear(d_model,d_model,bias=False)
        self.w_k=nn.Linear(d_model,d_model,bias=False)
        self.w_v=nn.Linear(d_model,d_model,bias=False)
        self.w_o=nn.Linear(d_model,d_model,bias=False)

    #Attention function in staticmethod (può essere chiamata fuori da un'istanza MultiheadAttention)
    @staticmethod
    def attention(query,key,value,mask,dropout: nn.Dropout):
        d_k=query.shape[-1]
        attention_scores=torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores=nn.Softmax(dim=-1)(attention_scores)
        if dropout is not None:
            attention_scores=dropout(attention_scores)
        return torch.matmul(attention_scores, value), attention_scores
        
    def forward(self,q,k,v,mask):
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)
        
        #Dividiamo ciascuna in h matrici che hanno ultima dimensione d_k
        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
        
        x, self.attention_scores=MultiHeadAttention.attention(query,key,value,mask,self.dropout)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
        return self.w_o(x)

#Skip connection
class ResidualConnection(nn.Module):
    def __init__(self,dropout: float):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNorm()
        
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

#Combiniamo gli ingredienti sopra nell'architettura dell'Encoder
class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block: MultiHeadAttention,feed_forward_block: FeedForwardBlock,dropout: float):
        super().__init__()
        self.s_a=self_attention_block
        self.f_f=feed_forward_block
        self.res_con=nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self,x,src_mask):
        x=self.res_con[0](x, lambda x: self.s_a(x,x,x,src_mask))
        x=self.res_con[1](x, lambda x: self.f_f(x))
        return x

#Iteriamo n volte il EncoderBlock
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNorm()
        
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)
    

#Strati per il modulo Decoder e strato finale (projection layer)
class DecoderBlock(nn.Module):
    def __init__(self,self_attention: MultiHeadAttention,cross_attention: MultiHeadAttention,feed_forward_block: FeedForwardBlock,dropout: float):
        super().__init__()
        self.self_attention=self_attention
        self.cross_attention=cross_attention
        self.f_f=feed_forward_block
        self.res_con=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self,x,y,src_mask,tgt_mask):
        x=self.res_con[0](x, lambda x: self.self_attention(x,x,x,tgt_mask))
        x=self.res_con[1](x, lambda x: self.cross_attention(x,y,y,src_mask))
        x=self.res_con[2](x, lambda x: self.f_f(x))
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNorm()
        
    def forward(self,x,y,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,y,src_mask,tgt_mask)
        return self.norm(x)

#Strato finale
class ProjectionLayer(nn.Module):
    def __init__(self,d_model: int,vocab_size: int):
        super().__init__()
        self.linear=nn.Linear(d_model,vocab_size)
        
    def forward(self,x):
        return nn.Softmax(dim=-1)(self.linear(x)) 
    

#Architettura Transformer
class Transformer(nn.Module):
    def __init__(self,src_embedding: InputEmbedding,tgt_embedding: InputEmbedding,src_position: PositionalEncoding,tgt_position: PositionalEncoding,encoder: Encoder,decoder: Decoder,projection: ProjectionLayer):
        super().__init__()
        self.src_emb=src_embedding
        self.tgt_emb=tgt_embedding
        self.src_pos=src_position
        self.tgt_pos=tgt_position
        self.encoder=encoder
        self.decoder=decoder
        self.proj=projection
        
    def encode(self,src,src_mask):
        src=self.src_emb(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)
        
    def decode(self,encoder_output,tgt,src_mask,tgt_mask):
        tgt=self.tgt_emb(tgt)
        tgt=self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
        
    def projection(self,x):
        return self.proj(x) 
    
#Funzione che mette insieme tutto
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, dropout: float=0.1,N: int=6, h: int=8) -> Transformer:
    #Embedding layers
    src_embedding=InputEmbedding(d_model,src_vocab_size)
    tgt_embedding=InputEmbedding(d_model,tgt_vocab_size)
    #Position Layers
    src_position=PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_position=PositionalEncoding(d_model,tgt_seq_len,dropout)
    #Encoder Layers
    blocks=[]
    for _ in range(N):
        self_a_block=MultiHeadAttention(d_model,h,dropout)
        feed_for_block=FeedForwardBlock(d_model,4*d_model,dropout)
        block=EncoderBlock(self_a_block,feed_for_block,dropout)
        blocks.append(block)
    #Decoder Layers
    blocks2=[]
    for _ in range(N):
        self_a_block=MultiHeadAttention(d_model,h,dropout)
        cross_a_block=MultiHeadAttention(d_model,h,dropout)
        feed_for_block=FeedForwardBlock(d_model,4*d_model,dropout)
        block=DecoderBlock(self_a_block,cross_a_block,feed_for_block,dropout)
        blocks2.append(block)
    #Costruiamo Encoder e Decoder
    encoder=Encoder(nn.ModuleList(blocks))
    decoder=Decoder(nn.ModuleList(blocks2))
    #Projection Layers
    projection=ProjectionLayer(d_model,tgt_vocab_size)
    #Facciamo il Transformer
    transformer=Transformer(src_embedding,tgt_embedding,src_position,tgt_position,encoder,decoder,projection)
    #Inizializzazione Parametri
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return transformer