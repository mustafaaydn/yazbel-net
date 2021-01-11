# Komut satırı argümanlarının parse edilip eğitim işlemlerinin yapıldığı yer
import argparse
import json
import logging
import os

from data_loader import make_dataset
from network import YazbelNet
from yazbel_parser import save_user_replies

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# Command line parser
parser = argparse.ArgumentParser(
                # let's show the defaults in --help too
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("username",
                    help="yazbel kullanıcı adı")

parser.add_argument("--driver-path",
                    help="chromedriver.exe'ye giden yol",
                    default="chromedriver")

parser.add_argument("--seq-length",
                    help="model kaç karakter geriye baksın?",
                    type=int,
                    default=100)

parser.add_argument("--batch-size",
                    help="bir alt dönüşte kaç numune işlensin?",
                    type=int,
                    default=4)

parser.add_argument("--embedding-dim",
                    help="kelimeler kaç boyutta temsil edilsin?",
                    type=int,
                    default=256)

parser.add_argument("--rnn-hidden-units",
                    help="RNN kaç adet gizli birim kullansın?",
                    type=int,
                    default=128)

parser.add_argument("--loss",
                    help="kayıp fonksiyonu ne olsun?",
                    default="sparse_categorical_crossentropy")

parser.add_argument("--optimizer",
                    help="eniyileştirici ne olsun?",
                    default="adam")

parser.add_argument("--epochs",
                    help="veri üzerinde *en fazla* kaç tam tur dönülsün?",
                    type=int,
                    default=20)

parser.add_argument("--val-frac",
                    help="verinin ne kadarlık fraksiyonu validasyona gitsin?",
                    type=float,
                    default=0.1)

parser.add_argument("--es-patience",
                    help="erken duruş için kaç tam tur sabredilsin?",
                    type=int,
                    default=5)

args = parser.parse_args()

# used many times, so assign it to a variable :)
username = args.username

# if found that already parsed replies exist for a username, ask if still want
# to parse again
to_parse = True
if os.path.exists(f"replies/{username}.txt"):
    ans = input(f"Saved replies found for `{username}` - do you still want"
                " to get replies from forum.yazbel.com? (y / [n]): ")
    to_parse = ans.lower().startswith("y")

# saved_models dir if not exists yet
os.makedirs("saved_models", exist_ok=True)

# path to save / load model
model_path = os.path.join("saved_models", f"{username}_model")

# If it's going to parse, it also means training will take place too; but
# reverse may not necessarily hold
to_train = to_parse

if to_parse:
    save_user_replies(username, args.driver_path)
    logging.info("Kullanıcının forumdaki yanıtları elde edildi ve kaydedildi")
elif os.path.exists(model_path + ".index"):
    # already found a trained model for this user
    from network import _check_model, _load_model
    model = _load_model(username, args, model_path)

    # check if configs are the same
    model_ok = _check_model(username, args)
    if model_ok:
        # ok, same configs; ask
        ans = input(f"Trained model found for `{username}` - do you still want"
                    " to train? (y / [n]): ")
        to_train = ans.lower().startswith("y")
    else:
        # configs are not the same!
        to_train = True
else:
    # rare case: it's not going to parse but no saved model found either
    to_train = True

if to_train:
    # prepare the dataset
    train_ds, val_ds, (char2num, num2char) = make_dataset(
                                                    username,
                                                    val_frac=args.val_frac,
                                                    seq_length=args.seq_length,
                                                    batch_size=args.batch_size
                                                )
    logging.info("Kullanıcının yanıtlarından veri seti oluşturuldu")

    # make the model
    logging.info("Model oluşturuluyor ve eğitim (training) başlıyor..")

    model = YazbelNet(vocab_size=len(char2num),
                      embedding_dim=args.embedding_dim,
                      rnn_hidden_units=args.rnn_hidden_units)
    # train the model (may take time! e.g. hours)
    model = model.train(train_ds, val_ds=val_ds, loss=args.loss,
                        optimizer=args.optimizer, epochs=args.epochs,
                        es_patience=args.es_patience)

    logging.info("Modelin eğitimi tamamlandı")

    # save the weights and configs
    model.save_weights(model_path)

    config_save_path = os.path.join("saved_models", f"{username}_config.txt")
    with open(config_save_path, "w") as fh:
        json.dump(vars(args), fh)

    logging.info("Model kaydedildi")
