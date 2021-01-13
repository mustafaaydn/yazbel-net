# Bir karakter verildiğinde sıradaki karakterin ne olması gerektiği üzerine
# "düşünen", RNN temelli ve karakter bazlı öğrenen ağa ait sınıf

import json
import os

import tensorflow as tf


class YazbelNet(tf.keras.Model):
    """
    RNN temelli, karakter-bazlı öğrenen ağ.
    """
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_units):
        """
        Parameters
        -----------
        vocab_size: int
            Öğrenilecek "dil"in sözlüğündeki özgün eleman (karakter) sayısı

        embedding_dim: int
            Karakterleri temsil etmek için onları sayısal bir biçime "gömeriz",
            bu da bu sayısal uzayın boyutunu söyler. Örneğin 40 ise, her bir
            karakter 40 boyutlu bir uzayda bir nokta şeklinde temsil edilir.
            Dahası, eğitim süresinde bu temsil de öğrenilir!

        rnn_hidden_units: int
            Ara kısımda kullanılan RNN'in (burada GRU kullanılıyor), gizli
            katmanındaki ünite sayısını belirler.
        """
        super().__init__(self)

        # store some attributes, helps in model loading
        self.embedding_dim = embedding_dim
        self.rnn_hidden_units = rnn_hidden_units

        # Katmanlar: gömme, RNN ve "dense". Sonuncusu RNN'den gelen
        # çıktıları tüm sözlük üzerinde bir olasılık dağılımına çevirir.
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.GRU(rnn_hidden_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        """
        İleri salınımın gerçekleştiği yer (forward propagation).

        Parameters
        -----------
        inputs: tf.Tensor
            Modele gelen girdiler

        states: tf.Tensor, opsiyonel, varsayılan=None
            modelin dahili "durumu". Bu parametre model eğitilirken
            kullanılmıyor, varoluş amacı tekst üretimi kısmına "stateful"luğu
            sağlamak. Ayrıntı için `text_generator.generate_one_step`in
            docstring'ine bakabilirsiniz.

        return_state: bool, opsiyonel, varsayılan=False
            Hakeza, bu parametre de `states` gibi eğitimde kullanılmaz.

        training: bool, opsiyonel, varsayılan=False
            Olur da ileride dropout / batch_norm felan koyarsak manalı hale
            gelir, şu anda önemsiz :/.

        Returns
        ---------
            modelin ileri salınım sonucu ürettiği çıktı, tf.Tensor tipinde.
        """
        x = inputs
        # evvela gömüyoruz
        x = self.embedding(x, training=training)

        # bu kısım tekst üretimi safhası için var, eğitimde yok
        if states is None:
            states = self.rnn.get_initial_state(x)

        # RNN çalışıyor
        x, states = self.rnn(x, initial_state=states, training=training)

        # RNN'in çıktısını sözlükteki elemanlar (karakter) üzerine bir nevi
        # "olasılık" dağılımı olacak şekilde projeksiyona uğratıyoruz
        x = self.dense(x, training=training)

        # tekrar: tekst üretimi ile ilgili
        if return_state:
            return x, states
        return x

    def train(self, train_ds, val_ds=None,
              loss="sparse_categorical_crossentropy", optimizer="adam",
              epochs=20, es_patience=5):
        """
        Modelin eğitilmesi prosedürünü bir araya toplayan fonksiyon

        Parameters
        -------------
        train_ds: tf.data.Dataset
            Eğitim için ayrılmış veriseti

        val_ds: tf.data.Dataset, opsiyonel, varsayılan=None
            Validasyon için kullanılacak olan veriseti. `None` ise validasyon
            yapılmaz, aşırı-öğrenme (ovefitting) gerçekleşebilir!

        loss: str veya tf.losses.Loss örneği, opsiyonel,
                        varsayılan="sparse_categorical_crossentropy"
            Kayıp fonksiyonu. Bir nevi sınıflandırma yapıyoruz, dolayısıyla
            bu amaca yönelik kayıp fonksiyonları tercih edilebilir.

        optimizer: str veya tf.optimizers.Optimizer örneği, opsiyonel,
                        varsayılan="adam"
            Kaybı daha aza indirgemek için kullanılan "eniyileştirici".

        epochs: int, opsiyonel, varsayılan=20
            `train_ds` üzerinde (en fazla) kaç defa tam tur geçilmesi
            gerektiğini belirtir.

        es_patience: int, opsiyonel, varsayılan=5
            Erken duruş (early stopping) için sabredilmesi gereken "epoch"
            yani tam tur sayısı. Ancak `val_ds` `None` değilse anlamlıdır.
        """
        # her ne kadar varsayılan olsa da, kendisinin `from_logist`
        # parametresinin varsayılan değeri bize uymuyor onu değiştirelim :)
        if loss == "sparse_categorical_crossentropy":
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                                                        from_logits=True)
        # erken duruş ayarı: aşırı-öğrenmeye birebir!
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=es_patience,
                                                      mode="min")
        # model derlenir ve "fit" edilir yani esas öğrenmenin merkezi burası
        self.compile(optimizer=optimizer, loss=loss)
        self.fit(train_ds, epochs=epochs, validation_data=val_ds, shuffle=True,
                 callbacks=[early_stop])

        return self


def _check_model(username, current_args):
    """
    Diskten yüklenen `model` şu anki çalıştırılmaya çalışılan modelle aynı
    konfigürasyona mı sahip?
    """
    config_save_path = os.path.join("saved_models", f"{username}_config.txt")
    with open(config_save_path, "r") as fh:
        old_config_dict = json.load(fh)
    return vars(current_args) == old_config_dict


def _load_model(username, net_args, model_path):
    """
    Diskten model yükler.
    """
    # get mappers
    from data_loader import _get_text, _prepare_mappers
    char2num, _ = _prepare_mappers(_get_text(username))

    # make model and load the weights to it
    model = YazbelNet(vocab_size=len(char2num),
                      embedding_dim=net_args.embedding_dim,
                      rnn_hidden_units=net_args.rnn_hidden_units)

    # expect partial because we will only use it for inference
    model.load_weights(model_path).expect_partial()

    return model
