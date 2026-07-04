"""SONAR text encoder/decoder wrapper.

Lazy-loads the SONAR encoder (question/text -> embedding) and decoder
(embedding -> text) from their public Hugging Face repos. Both are frozen;
Atlas never fine-tunes SONAR.

Ported behavior-for-behavior from the validated dev inference script
(`atlas_run.py`): attention-masked mean pooling for encoding, and
encoder-output injection + beam search for decoding. Do not change the
pooling or generation settings — the released Atlas weights were validated
against exactly these.
"""

from __future__ import annotations

import torch

DEFAULT_ENCODER = "cointegrated/SONAR_200_text_encoder"
DEFAULT_DECODER = "raxtemur/SONAR_200_text_decoder"
DEFAULT_LANG = "eng_Latn"


class Sonar:
    """Lazy, device-aware SONAR encode/decode.

    Encoder and decoder are loaded on first use (each is >1 GB), so e.g. a
    retrieval-only session (``decode=False``) never pays for the decoder.
    """

    def __init__(
        self,
        encoder_model: str = DEFAULT_ENCODER,
        decoder_model: str = DEFAULT_DECODER,
        device: torch.device | str = "cpu",
        lang_code: str = DEFAULT_LANG,
    ):
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.device = torch.device(device)
        self.lang_code = lang_code

        self._encoder = None
        self._enc_tok = None
        self._decoder = None
        self._dec_tok = None

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _load_encoder(self):
        if self._encoder is None:
            from transformers import AutoTokenizer
            from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

            print(f"  Loading SONAR encoder ({self.encoder_model})...")
            self._enc_tok = AutoTokenizer.from_pretrained(self.encoder_model)
            self._enc_tok.src_lang = self.lang_code
            self._encoder = (
                M2M100Encoder.from_pretrained(self.encoder_model)
                .to(self.device)
                .eval()
            )
            for p in self._encoder.parameters():
                p.requires_grad = False
        return self._encoder, self._enc_tok

    @torch.no_grad()
    def encode(self, text: str, max_length: int = 64) -> torch.Tensor:
        """Encode a single text string to a SONAR embedding ``[1, D]``.

        Attention-masked mean pooling over the encoder's last hidden state —
        identical to how the Atlas fact pool and training data were encoded.
        """
        encoder, tok = self._load_encoder()
        batch = tok(
            [text], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(self.device)
        out = encoder(**batch)
        mask = batch.attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled.clone()

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def _load_decoder(self):
        if self._decoder is None:
            from transformers import M2M100ForConditionalGeneration, NllbTokenizer

            print(f"  Loading SONAR decoder ({self.decoder_model})...")
            self._dec_tok = NllbTokenizer.from_pretrained(self.decoder_model)
            self._decoder = (
                M2M100ForConditionalGeneration.from_pretrained(self.decoder_model)
                .to(self.device)
                .eval()
            )
            for p in self._decoder.parameters():
                p.requires_grad = False
        return self._decoder, self._dec_tok

    @torch.no_grad()
    def decode(
        self,
        emb: torch.Tensor,
        max_length: int = 64,
        num_beams: int = 4,
    ) -> str:
        """Decode a SONAR embedding (``[D]`` or ``[1, D]``) back to text.

        The embedding is injected as a single-position encoder output and
        decoded with beam search, forced to ``lang_code``.
        """
        decoder, tok = self._load_decoder()
        from transformers.modeling_outputs import BaseModelOutput

        pooled = emb.unsqueeze(0) if emb.dim() == 1 else emb
        enc_out_seq = pooled.to(self.device).unsqueeze(1)
        encoder_outputs = BaseModelOutput(last_hidden_state=enc_out_seq)
        forced_bos = tok.convert_tokens_to_ids(self.lang_code)
        out_ids = decoder.generate(
            encoder_outputs=encoder_outputs,
            forced_bos_token_id=forced_bos,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
        return tok.batch_decode(out_ids, skip_special_tokens=True)[0]
