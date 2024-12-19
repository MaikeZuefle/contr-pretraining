from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import regex as re
import math
import time
# based on https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html



@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


class WordsBoundaryFinder:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model() #.to(self.device)
        self.labels = self.bundle.get_labels()
        self.dictionary = {c: i for i, c in enumerate(self.labels)}

    def get_word_tensors(self, audio, transcript, start=0, end=None, tokenizer=None):
        

        waveform, sample_rate  =  audio
        words_with_boundaries = self.find_words_boundary([waveform], [transcript],  [start], [end], tokenizer)
        segments = []
        for w in words_with_boundaries[0]:

            start_sample = int(w[0] *sample_rate)
            end_sample = int(w[1]*sample_rate)

            segment = waveform[0, start_sample:end_sample].unsqueeze(0)
            segments.append((segment, w[2]))
        return segments

    def find_words_boundary(self, audios, transcripts, starts=None, ends=None, tokenizer=None, rank=None):
        # pad audios

        waveforms = [tensor.squeeze(0) for tensor in audios]
        padded_waveforms = pad_sequence(waveforms, batch_first=True)
        original_lengths = [waveform.size(0) for waveform in waveforms]
        padding_mask = torch.arange(padded_waveforms.size(1))[None, :] < torch.tensor(original_lengths)[:, None]
        

        if rank:
            device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        with torch.inference_mode():
            self.model.eval()
            if rank:
                self.model = self.model.to(device)
                batch_emissions, _ = self.model(padded_waveforms.to(device))
            else:
                self.model = self.model.to(self.device)
                batch_emissions, _ = self.model(padded_waveforms.to(self.device))

            batch_emissions = torch.log_softmax(batch_emissions, dim=-1)

        # adjust padding mask to new shape of emissions
        max_length = max(original_lengths)
        batch_size, time_steps, num_classes = batch_emissions.shape
        downsampling_factor = max_length / time_steps  # Assume uniform downsampling for each waveform
        adjusted_padding_mask = padding_mask[:, :time_steps * int(downsampling_factor):int(downsampling_factor)]
        assert adjusted_padding_mask.size(1) == time_steps, f"Adjusted padding mask shape {adjusted_padding_mask.size(1)} does not match emissions time dimension {time_steps}"



        unpadded_emissions = []

        for i in range(batch_size):
            # Calculate the effective length in the downsampled space
            effective_length = math.ceil(original_lengths[i] / downsampling_factor)
            unpadded_emission = batch_emissions[i, :effective_length, :]
            unpadded_emissions.append(unpadded_emission.unsqueeze(0))


        batched_word_with_boundaries = []
        for idx in range(len(audios)):
            
            audio = audios[idx]
            emissions = unpadded_emissions[idx]
            transcript = transcripts[idx]
            end = ends[idx]
            start = starts[idx]


            if tokenizer:
                transcript_words = tokenizer.tokenize(transcript)
                transcript_words_adapted = [token.replace('Ġ', '').replace('Ċ', '') for token in transcript_words]
                transcript_words_adapted = [self._adapt_word(t) for t in transcript_words_adapted]
            else:
                transcript_words = transcript.split()
                transcript_words_adapted = [self._adapt_word(t) for t in transcript_words]

            transcript_words_wo_empty = [
                transcript_word
                for transcript_word in transcript_words_adapted
                if transcript_word != ""
            ]
            if len(transcript_words_wo_empty) == 0:
                
                words_with_boundaries = []
                duration_per_word = (end - start) / len(transcript_words)
                for i, transcript_word in enumerate(transcript_words):
                    words_with_boundaries.append(
                        (
                            start + i * duration_per_word,
                            start + (i + 1) * duration_per_word,
                            transcript_word,
                        )
                    )

                batched_word_with_boundaries.append(words_with_boundaries)
                continue



            transcript = '|' + '|'.join(transcript_words_adapted) + '|'        

            # Detaching emission
            emission = emissions[0].cpu().detach()

            
            # trellis and path calculation
            tokens = [self.dictionary[c] for c in transcript]

            trellis = self._get_trellis(emission, tokens)
            path = self._backtrack(trellis, emission, tokens)
            segments = self._merge_repeats(path, transcript)

            # word merging
            ratio = audio.size(1) / trellis.size(0)
            word_segments = self._merge_words(segments)
            word_segments_with_boundaries = self._compute_word_segments_with_boundaries(word_segments, ratio, start)
            words_with_boundaries = self._compute_words_with_boundaries(transcript_words, transcript_words_adapted, word_segments_with_boundaries)
            words_with_boundaries = self._add_whitespace_before_word(words_with_boundaries)
            batched_word_with_boundaries.append(words_with_boundaries)

        return batched_word_with_boundaries


    def _add_whitespace_before_word(self, tokens):

        adjusted_tokens = [(0, tokens[0][1], tokens[0][2])]  # first token starts at 0

        for i in range(1, len(tokens)):
            prev_end = adjusted_tokens[-1][1]
            current_start, current_end, token = tokens[i]

            # Adjust the start time by adding the gap to the current start time
            new_start = prev_end

            adjusted_tokens.append((new_start, current_end, token))
        return adjusted_tokens


    def _adapt_word(self, word):
        word = word.upper()
        word = word.replace('Ä', 'AE').replace('Ö', 'OE').replace('Ü', 'UE')
        word = word.replace('"', "'").replace('´', "'").replace('`', "'")
        word = re.sub("[^A-Z'\|]", '', word)
        word = word.strip()
        return word

    def _get_trellis(self, emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)


        trellis = torch.zeros((num_frame, num_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[-num_tokens + 1:, 0] = float("inf")

        for t in range(num_frame - 1):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens[1:]],
            )


        return trellis

    def _backtrack(self, trellis, emission, tokens, blank_id=0):
        t, j = trellis.size(0) - 1, trellis.size(1) - 1

        path = [Point(j, t, emission[t, blank_id].exp().item())]

        while j > 0:
            # Should not happen but just in case
            assert t > 0

            # 1. Figure out if the current position was stay or change
            # Frame-wise score of stay vs change
            p_stay = emission[t - 1, blank_id]
            p_change = emission[t - 1, tokens[j]]

            # Context-aware score for stay vs change
            stayed = trellis[t - 1, j] + p_stay
            changed = trellis[t - 1, j - 1] + p_change

            # Update position
            t -= 1
            if changed > stayed:
                j -= 1

            # Store the path with frame-wise probability.
            prob = (p_change if changed > stayed else p_stay).exp().item()
            path.append(Point(j, t, prob))

        # Now j == 0, which means, it reached the SoS.
        # Fill up the rest for the sake of visualization
        while t > 0:
            prob = emission[t - 1, blank_id].exp().item()
            path.append(Point(j, t - 1, prob))
            t -= 1

        return path[::-1]

    def _merge_repeats(self, path, transcript):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments

    def _merge_words(self, segments, separator="|"):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words

    def _compute_word_segments_with_boundaries(self, word_segments, ratio, start):
        word_segments_with_boundaries = []
        for word_segment_index, word in enumerate(word_segments):
            word = word_segments[word_segment_index]
            x0 = int(ratio * word.start)
            x1 = int(ratio * word.end)
            word_start = x0 / self.bundle.sample_rate + start
            word_end = x1 / self.bundle.sample_rate + start
            word_segments_with_boundaries.append((word_start, word_end, word.label))
            #print(f"{word.label} {word_start:.3f} - {word_end:.3f}")

        return word_segments_with_boundaries

    def _compute_words_with_boundaries(self, transcript_words, transcript_words_adapted, word_segments_with_boundaries):
        words_with_boundaries = []
        transcript_words_adapted_index = 0
        output_words_index = 0
        while transcript_words_adapted_index < len(transcript_words_adapted):

            if transcript_words_adapted[transcript_words_adapted_index] == '':
                number_of_empty_words = 1
                while (transcript_words_adapted_index + number_of_empty_words < len(transcript_words_adapted) and
                       transcript_words_adapted[transcript_words_adapted_index + number_of_empty_words] == ''):
                    number_of_empty_words += 1

                start = (word_segments_with_boundaries[output_words_index - 1][1]
                         if output_words_index > 0
                         else word_segments_with_boundaries[output_words_index][0])
                end = (word_segments_with_boundaries[output_words_index][0]
                       if output_words_index < len(word_segments_with_boundaries)
                       else word_segments_with_boundaries[-1][1])
                duration_per_word = (end - start) / number_of_empty_words

                for i in range(number_of_empty_words):
                    start_transcript_word = start + i * duration_per_word
                    end_transcript_word = start + (i + 1) * duration_per_word
                    words_with_boundaries.append((start_transcript_word, end_transcript_word,
                                                  transcript_words[transcript_words_adapted_index]))
                    transcript_words_adapted_index += 1
            else:
                start, end, _ = word_segments_with_boundaries[output_words_index]
                words_with_boundaries.append((start, end, transcript_words[transcript_words_adapted_index]))
                output_words_index += 1
                transcript_words_adapted_index += 1

        return words_with_boundaries

