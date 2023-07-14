import tiktoken

sentence = "hello world"

print("Sentence/Word: ", sentence)

print("********** Encoding based on cl100k_base encoder **********")

# First run requires internet connection.
enc_name = tiktoken.get_encoding("cl100k_base")
token_ids = enc_name.encode(sentence)
if token_ids:
    print("Number of tokens is/are: ", len(token_ids))


model_name = "gpt-3.5-turbo-0613"

print("********** Encoding based on model's default encoder **********")
print("********** Model used is " + model_name + " **********")

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc_model = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
token_ids = enc_model.encode(sentence)
if token_ids:
    print("Number of tokens is/are: ", len(token_ids))
