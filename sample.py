# Bir model `train.py` aracılığıyla eğitildikten sonra tekst üretimi burada
# CLI ile gerçekleşir

import argparse
import json
import logging
import os

from data_loader import _get_text, _prepare_mappers
from network import _load_model
from text_generator import TextGenerator

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# Command line parser
parser = argparse.ArgumentParser(
                # let's show the defaults in --help too
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("username",
                    help="yazbel kullanıcı adı")

parser.add_argument("--length",
                    help="üretilen tekst kaç karakter uzunluğunda olsun?",
                    type=int,
                    default=200)

parser.add_argument("--seed",
                    help="model cümleye hangi kelime ile başlasın?",
                    default="Merhaba")

parser.add_argument("--temperature",
                    help="üretilen tekstin harareti kaç olsun?",
                    type=float,
                    default=0.5)
args = parser.parse_args()

username = args.username
model_path = os.path.join("saved_models", f"{username}_model")
if not os.path.exists(model_path + ".index"):
    logging.error("You need to train the model first and then sample!")
else:
    logging.info("Model bulundu, yükleniyor..")

    # # get configs first
    config_save_path = os.path.join("saved_models", f"{username}_config.txt")
    with open(config_save_path, "r") as fh:
        config_dict = json.load(fh)
    args.embedding_dim = config_dict["embedding_dim"]
    args.rnn_hidden_units = config_dict["rnn_hidden_units"]

    # load the model
    model = _load_model(username, args, model_path)

    logging.info("Model diskten yüklendi")

    # get the mappers
    char2num, num2char = _prepare_mappers(_get_text(username))

    # text generation!
    logging.info("Metin üretiliyor..")

    gen = TextGenerator(model, char2num, num2char,
                        temperature=args.temperature)
    generated_text = gen.sample_text(length=args.length, seed=args.seed)

    print("Üretilen metin:", end="\n"+"-"*40+"\n")
    print(generated_text)
