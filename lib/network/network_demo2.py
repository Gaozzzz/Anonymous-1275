import torch
import torch.nn as nn

from .module import mit_1100, Encoder, Decoder, Pred


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mit_1100.mit_b4()
        self._init_weights()  # load pretrain
        self.Encoder = Encoder.Encoder(self.backbone.embed_dims)  # embed_dims=[64, 128, 320, 512]
        self.Cl_Encoder = Encoder.Cl_Encoder(self.backbone.embed_dims)
        self.Decoder = Decoder.Decoder(self.backbone.embed_dims)
        self.Out_Pred = Pred.Out_Pred()

    def _init_weights(self):
        pretrained_dict = torch.load('lib/backbone/mit_b4.pth')
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("Successfully load pretrain model -- mit_b4 for ICFormer")

    def forward(self, x):
        backbone = self.backbone(x)
        encoder = self.Encoder(backbone)  # encoder = [encoder1, encoder2, encoder3, encoder4]
        encoder, contrast_loss = self.Cl_Encoder(encoder)  # encoder = [encoder1, encoder2, cl_encoder3, cl_encoder4]
        decoder = self.Decoder(encoder)  # decoder = [decoder1, decoder2, decoder3, decoder4]
        pred = self.Out_Pred(decoder)
        res = pred, contrast_loss

        """out"""
        return res


if __name__ == '__main__':
    ras = Network().cpu()
    input_tensor = torch.randn(16, 3, 352, 352).cpu()
    out = ras(input_tensor)
    print(out)
