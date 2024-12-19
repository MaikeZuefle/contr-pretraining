from collections import defaultdict
import json
import datasets
from datasets import Dataset

def get_unique_questions(data):
    conversation_dict = defaultdict(list)
    # Collect conversations and create an index mapping
    for i, conversation in enumerate(data['conversations']):
        for message in conversation:
            if message['from'] == 'human':
                human_message = message['value']
            elif message['from'] == 'gpt':
                gpt_response = message['value']
        

        conversation_dict[human_message].append(gpt_response)
    return conversation_dict

def update_gpt_values(example, new_conversations):
    for conversation in example['conversations']:
        if conversation['from'] == 'human':
            human_value = conversation['value']
        elif conversation['from'] == 'gpt':
            conversation['value'] = json.dumps(new_conversations.get(human_value, conversation['value']))
    return example

def process_conversations(data):
    new_conversations = get_unique_questions(data)
    updated_dataset = data.map(lambda example: update_gpt_values(example, new_conversations), batched=False)

    # convert to pandas to get indies of unique entries (as datasets does not support that)
    df = updated_dataset.to_pandas()
    df['conversations_tuple'] = df['conversations'].apply(lambda x: tuple(tuple(d.items()) for d in x))
    df_unique = df.drop_duplicates(subset=['conversations_tuple']).drop(columns=['conversations_tuple']).index.tolist()
    # use indices to retrieve unique data
    updated_dataset = updated_dataset.select(df_unique)
    return updated_dataset