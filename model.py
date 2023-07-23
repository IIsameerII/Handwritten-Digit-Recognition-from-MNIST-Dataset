import torch
# Import nn submodule to subclass our neural network
from torch import nn

def initialize_model(Pytorch_file_path):

    """Initializes the TinyVGG Model for Inference

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        file_dir: An open smalltable.Table instance.
        model_filename: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.

    Returns:
        A pre-trained TinyVGG PyTorch model for inference

    Raises:
        FileNotFound: An error occurred accessing the directory.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class TinyVGG(nn.Module):
        def __init__(self,
                    in_features,
                    out_features,
                    hidden_units):
            super().__init__()
            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(in_channels=in_features,
                                    out_channels=hidden_units,
                                    kernel_size=3,
                                    padding=1,
                                    stride=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=2,
                        padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                # We need to change this in_features below
                nn.Linear(in_features=7*7*hidden_units,  # This is a hardcoded value. 
                                            #The error in the dummy_x gives us the info for this
                        out_features=out_features)
            )

        def forward(self, X):
            X = self.conv_block_1(X)
            X = self.conv_block_2(X)
            X = self.classifier(X)
            return (X)

    model = TinyVGG(in_features=1,
                        out_features=10,
                        hidden_units=10).to(device)
    
    model.load_state_dict(torch.load(f=Pytorch_file_path),map_location=torch.device(device))
    
    return model
                                    