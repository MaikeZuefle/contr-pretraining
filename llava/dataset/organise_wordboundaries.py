import json
import random

random.seed(42)
from copy import deepcopy


def distribute_uniform(words, tokenized_text, sr):
    result = {"subword_indices": []}

    base_tokens_per_tensor = len(tokenized_text) // len(words)
    extra_tokens = len(tokenized_text) % len(words)
    subword_indices = []
    start_index = 0

    for i in range(len(words)):
        # Determine the end index for each tensor
        end_index = start_index + base_tokens_per_tensor
        if i < extra_tokens:
            end_index += 1  # Distribute the extra tokens evenly

        # Append the subwords to the subword_indices list
        subword_indices.append(tokenized_text[start_index:end_index])

        # Update the start index for the next tensor
        start_index = end_index

    result["subword_indices"] = subword_indices
    return result


def match_subwords_to_words(words, tokenized_text, sr):
    tokenized_text = [t if t else "[PAD]" for t in tokenized_text]
    result = []
    token_index = 0
    result = {"subword_indices": []}
    unmatched_things = []

    for word_idx, word in enumerate(words):

        subwords = []
        subword_indices = []
        combined_subword = ""
        matches = [t in word for t in tokenized_text[token_index:]]
        skip_word = all([not m for m in matches])
        if skip_word:
            result["subword_indices"].append(subword_indices)
            if word_idx == len(words) - 1:
                # if this is the last word, add everything
                unmatched_things += range(token_index, len(tokenized_text))
            else:
                unmatched_things.append(token_index)
            token_index += 1

            continue

        else:
            for m in matches:
                if m:
                    # if things could not be matched before but now there is a match, they should be stored in the previous list
                    if unmatched_things != []:
                        empty_slots = [
                            i
                            for i in range(word_idx)
                            if result["subword_indices"][i] == []
                        ]
                        if len(empty_slots) > 0:
                            tokens_per_slot = len(unmatched_things) // len(
                                empty_slots
                            )
                            remainder = len(unmatched_things) % len(
                                empty_slots
                            )
                            # Distribute the tokens across the slots
                            for i, slot in enumerate(empty_slots):
                                # Each slot gets the base number of tokens
                                result["subword_indices"][slot] = (
                                    unmatched_things[:tokens_per_slot]
                                )
                                unmatched_things = unmatched_things[
                                    tokens_per_slot:
                                ]

                                # The last 'remainder' slots get one extra token
                                if i >= len(empty_slots) - remainder:
                                    result["subword_indices"][slot].append(
                                        unmatched_things.pop(0)
                                    )
                        else:
                            # no empty slots so just put it in the current one
                            subword_indices += unmatched_things

                        unmatched_things = []
                    break
                else:
                    unmatched_things.append(token_index)
                    token_index += 1

        while (
            token_index < len(tokenized_text)
            and combined_subword != word
            and len(combined_subword) <= len(word)
        ):
            subword = tokenized_text[token_index]
            subwords.append(subword)
            subword_indices.append(token_index)
            combined_subword += subword
            token_index += 1
        result["subword_indices"].append(subword_indices)

    # if last things could not be matched, then they get appended to the last word
    if unmatched_things != []:
        # Find all remaining empty slots in subword_indices
        empty_slots = [
            i
            for i in range(len(result["subword_indices"]))
            if result["subword_indices"][i] == []
        ]

        # Calculate the base number of tokens per slot and the remainder
        tokens_per_slot = len(unmatched_things) // len(empty_slots)
        remainder = len(unmatched_things) % len(empty_slots)

        # Distribute the tokens across the slots
        for i, slot in enumerate(empty_slots):
            # Each slot gets the base number of tokens
            result["subword_indices"][slot] = unmatched_things[
                :tokens_per_slot
            ]
            unmatched_things = unmatched_things[tokens_per_slot:]

            # The last 'remainder' slots get one extra token
            if i >= len(empty_slots) - remainder:
                result["subword_indices"][slot].append(unmatched_things.pop(0))

    # in case any subwords at the end that have not been matched yet
    last_entry = (
        result["subword_indices"][-1][-1]
        if result["subword_indices"][-1] != []
        else None
    )
    if last_entry:
        if last_entry != len(tokenized_text) - 1:
            missing_subwords = [
                i for i in range(len(tokenized_text)) if i > last_entry
            ]
            result["subword_indices"][-1] += missing_subwords

    # should not happen but if something went wrong, then we just distribute uniformly
    if not last_entry or (
        result["subword_indices"][-1][-1] != len(tokenized_text) - 1
    ):
        print(
            f"Error in:\n\n {tokenized_text}, {result['subword_indices']}, {words}"
        )
        result = distribute_uniform(
            words, list(range(len(tokenized_text))), sr
        )

    return result


def split_into_text_speech(
    words, text_min=4, text_max=10, speech_min=2, speech_max=5
):
    total_words = len(words)
    result = []
    current_position = 0
    total_text_words = 0
    total_speech_words = 0

    # Randomly decide whether to start with 'Text' or 'Speech'
    start_with_text = random.choice([True, False])

    while current_position < total_words:
        remaining_words = total_words - current_position

        cur_min1 = text_min if start_with_text else speech_min
        cur_max1 = text_max if start_with_text else speech_max
        cur_min2 = text_min if not start_with_text else speech_min
        cur_max2 = text_max if not start_with_text else speech_max

        if remaining_words < text_min + speech_min:
            # If we don't have enough words to create both a Text and a Speech segment
            # Combine remaining words into one final segment based on the previous segment type
            if result and result[-1] == "Text":
                # If the last segment was Text, add remaining words to Speech
                result.extend(["Speech"] * remaining_words)
                total_speech_words += remaining_words
            else:
                # If the last segment was Speech or no previous segment, add remaining words to Text
                result.extend(["Text"] * remaining_words)
                total_text_words += remaining_words
            break
        else:
            span_size_1 = random.randint(
                cur_min1, min(cur_max1, remaining_words)
            )

            # Ensure speech span can be assigned within limits
            max_span_2 = min(cur_max2, remaining_words - span_size_1)
            span_size_2 = (
                random.randint(cur_min2, max_span_2)
                if max_span_2 >= cur_max2
                else max_span_2
            )

        text_span_size = span_size_1 if start_with_text else span_size_2
        speech_span_size = span_size_1 if not start_with_text else span_size_2

        # Add the current span to the result
        if start_with_text and text_span_size > 0:
            result.extend(["Text"] * text_span_size)
            current_position += text_span_size
            total_text_words += text_span_size
            start_with_text = False  # Alternate to speech next
        elif not start_with_text and speech_span_size > 0:
            result.extend(["Speech"] * speech_span_size)
            current_position += speech_span_size
            total_speech_words += speech_span_size
            start_with_text = True  # Alternate to text next

    return result


def merge_based_on_labels(subword_list, label_list, timed_word_list):

    assert len(subword_list) == len(label_list) == len(timed_word_list)
    if (
        not subword_list
        or not label_list
        or len(subword_list) != len(label_list)
    ):
        raise Exception

    merged_list = []
    merged_labels = []
    merged_timed_words = []

    current_group = deepcopy(subword_list[0])  # Start with the first group
    current_label = deepcopy(label_list[0])  # Track the current label
    current_timed_group = [deepcopy(timed_word_list[0])]

    # Iterate over the data and labels
    for i in range(1, len(subword_list)):
        subwords = deepcopy(subword_list[i])
        timed_words = deepcopy(timed_word_list[i])
        if label_list[i] == current_label:

            # If the label is the same, extend the current group
            current_group.extend(subwords)
            current_timed_group.extend([timed_words])
        else:
            # If the label changes, append the current group and label to the merged list
            merged_list.append(current_group)
            merged_labels.append(current_label)

            # Merge the timed words into a single span
            start_time = current_timed_group[0][0]
            end_time = current_timed_group[-1][1]

            merged_text = " ".join([word[2] for word in current_timed_group])
            merged_timed_words.append([start_time, end_time, merged_text])

            # Start a new group with the new label
            current_group = subwords
            current_timed_group = [timed_words]
            current_label = deepcopy(label_list[i])

    # Append the last group and label
    merged_list.append(current_group)
    merged_labels.append(current_label)

    # Merge the last timed word group
    start_time = current_timed_group[0][0]
    end_time = current_timed_group[-1][1]
    merged_text = " ".join([word[2] for word in current_timed_group])
    merged_timed_words.append([start_time, end_time, merged_text])

    return merged_list, merged_labels, merged_timed_words


def organise_word_boundaries(data, audio, sr, tokenizer):

    words_with_boundaries = json.loads(data["word_boundaries"])
    words = [w[2] for w in words_with_boundaries]

    tokenized_text = tokenizer.tokenize(data["transcription"])

    tokenized_text = [
        tokenizer.convert_tokens_to_string([t]).strip() for t in tokenized_text
    ]

    # match the subwords and words
    result = match_subwords_to_words(words, tokenized_text, sr)
    # get speech and text distrution for mixed NWP

    speech_text_labels = split_into_text_speech(result["subword_indices"])

    assert len(speech_text_labels) == (len(words))
    # apply the speech and text labels to get the time spans for the speech parts
    merged_subwords, merged_labels, merged_timed_words = merge_based_on_labels(
        result["subword_indices"], speech_text_labels, words_with_boundaries
    )
    assert (
        len(merged_subwords) == len(merged_labels) == len(merged_timed_words)
    )

    final_data = {
        "tensors": [],
        "subword_indices": [],
        "speech_text_label": [],
    }
    for i in range(len(merged_timed_words)):
        w = merged_timed_words[i]
        start_sample = int(w[0] * sr)
        end_sample = int(w[1] * sr)
        if end_sample - start_sample == 0:
            # this can happen if ther are digits that are not recognized by the aligner
            if i == len(merged_timed_words) - 1:
                # if it is at the end, we just give it the rest of the audio
                end_sample = len(audio)

            else:
                next_w_start = int(merged_timed_words[i + 1][0] * sr)
                next_w_end = int(merged_timed_words[i + 1][1] * sr)
                if next_w_start > end_sample:
                    # if there is still time until the next word begins, use this
                    end_sample = next_w_start
                else:
                    # otherwise just add the end of the next word
                    end_sample = next_w_end

        segment = audio[start_sample:end_sample].unsqueeze(0)

        final_data["tensors"].append((segment.squeeze(), sr))
        final_data["subword_indices"].append(merged_subwords[i])
        final_data["speech_text_label"].append(merged_labels[i])

    assert (
        final_data["subword_indices"][-1][-1] == len(tokenized_text) - 1
    ), print(
        final_data["subword_indices"],
        final_data["speech_text_label"],
        tokenized_text,
    )
    flattened_subwords = [
        item for sublist in final_data["subword_indices"] for item in sublist
    ]
    assert len(flattened_subwords) == len(set(flattened_subwords)), print(
        final_data["subword_indices"],
        final_data["speech_text_label"],
        tokenized_text,
    )
    return final_data
