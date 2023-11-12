import argparse

def parser(description = "Argument Parser for Predict.py"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', action="store", default="./flowers/", type = str)
    parser.add_argument('checkpoint', default='./checkpoint.pth', action="store", type = str)
    parser.add_argument('--top_k', dest = 'top_k', default=5, type=int)
    parser.add_argument('--category_names', default='cat_to_name.json', type = str)
    parser.add_argument('--gpu', default = 'cpu', type = str)
                        
    args = parser.parse_args ()
                        
    return args                    
        