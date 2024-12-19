from llava.dataset.custom_dataset.giga_nwp import Giga_NWP
from llava.dataset.custom_dataset.gigaspeech import Gigaspeech
from llava.dataset.custom_dataset.mls import MLS
from llava.dataset.custom_dataset.mustc import MuSTC
from llava.dataset.custom_dataset.mustc_nwp import MuSTC_NWP
from llava.dataset.custom_dataset.spoken_squad import Spoken_SQuAD


DATASET_MAPPING = {
    "MuSTC": MuSTC,
    "Spoken_SQuAD": Spoken_SQuAD,
    "MuSTC_NWP": MuSTC_NWP,
    "Gigaspeech": Gigaspeech,
    "MLS": MLS,
    "Giga_NWP": Giga_NWP,
}
