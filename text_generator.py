# Eğitilmiş bir YazbelNet üzerinden tekst üretimi yapmaya olanak sağlayan
# işlevleri barındıran sınıfın olduğu yer.
import logging

import numpy as np
import tensorflow as tf

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


class TextGenerator(tf.keras.Model):
    """
    Eğitilmiş YazbelNet örneği üzerinden tekst üretimi yapmaya olanak sağlar.
    """
    def __init__(self, model, char2num, num2char, temperature=1.):
        """
        Parameters
        -----------
        model: network.YazbelNet
            Halihazırda eğitilmiş olduğu varsayılan YazbelNet örneği

        char2num: dict
            Karakterleri tamsayılara eşleyen sözlük

        num2char: dict
            Tamsayıları karakterlere eşleyen sözlük (char2num'un tersi ama
            birçok yerde kullanıldığı için bir kere üretilip etrafa paslanıyor)

        temperature: float, optional, default=1.
            Tekst üreticinin "harareti". Pozitif bir float olan bu sayı,
            her bir adımda bir sonraki karakter seçilirken modelin ne kadar
            risk alması gerektiğini belirtiyor. Örneğin 0'a çok yakınken (0.01)
            en olası karakter seçiliyor ve dolayısıyla gramatik olarak nispeten
            daha doğru cümleler ortaya çıkıyor. Arttığında ise karakter seçimi
            içerisine rastgelelik girmeye başlıyor ve modelin değişik anlam
            yollarına sapması gözlenebiliyor.
        """
        super().__init__(self)
        self.char2num = char2num
        self.num2char = num2char
        self.temperature = temperature
        self.model = model

    @tf.function
    def generate_one_step(self, inputs, states=None):
        """
        Forwards the YazbelNet model for the inputs and gets the logits to
        return the next predicted character upon temperature.

        Parameters
        ----------
        inputs: tf.Tensor
            The input to be feed-forwarded to the `self.model`

        states: tf.Tensor, optional, default=None
            this is to preserve "stateful"ness of the text generation pipeline
            i.e. the last state of the RNN (if any) from the previous
            prediction is preserved and passed to the next one (unless we are
            at the very first step i.e. None case, in which case the underlying
            model will default to `self.model.rnn.initial_state` which is
            possibly 0).

        Returns
        -------
        predicted character (in numeric form) and the internal state of the rnn
        """
        # feedforward the input
        logits, states = self.model(inputs=inputs, states=states,
                                    return_state=True)
        # take the last timestep's values as logits and apply temperature
        logits = logits[:, -1, :]
        logits /= self.temperature

        # randomly sample the next character (its numeric mapping, actually)
        # from logits, but not uniform, weighted!
        predicted_num = tf.random.categorical(logits, num_samples=1)

        # return states too so that "stateful" text generation happens
        return predicted_num, states

    def sample_text(self, length=200, seed="Merhaba"):
        """
        Samples a `length` length text starting with `seed`.

        Parameters
        ----------
        length: int, optional, default=200
            The total number of characters to generate sequentially to form
            the text

        seed: str, optional, default="Merhaba"
            The text fires with this word.

        Returns
        --------
        The generated text

        Notes
        -----
        If any character of the given seed is not in the user's vocabulary,
        then we set the seed to a random letter from the vocabulary.
        """
        if any(char not in self.char2num for char in seed):
            logging.warn(f"The seed {seed} contains non-vocab characters,"
                         " defaulting to a random character..")
            seed = np.random.choice(np.array(list(self.char2num.keys())))

        # start with the seed
        next_seq = seed
        result = [next_seq]

        states = None
        for n in range(length):
            # convert the character(s) to numerics
            next_seq = np.array(
                            [self.char2num[char] for char in next_seq]
                        ).reshape(1, -1)
            # Get the next character prediction in numeric and convert to char
            # note that this will be the input to the model in next turn!
            next_num, states = self.generate_one_step(next_seq, states=states)
            next_seq = self.num2char[next_num.numpy().item()]

            result.append(next_seq)

        return "".join(result)
