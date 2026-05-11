import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from llm_helper import llm


def clean_text(text: str) -> str:
    """Remove or repair invalid Unicode surrogate characters."""
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")


def process_post(raw_file_path, processed_file_path="data/processed_posts.json"):
    enriched_post = []

    with open(raw_file_path, encoding="utf-8", errors="ignore") as file:
        posts = json.load(file, strict=False)
        for post in posts:
            # Clean original text field
            post["text"] = clean_text(post["text"])
            metadata = extract_metadata(post["text"])
            post_with_metadata = post | metadata
            enriched_post.append(post_with_metadata)

    # Print enriched posts
    for epost in enriched_post:
        print(epost)

    # Clean all data before saving
    def clean_obj(obj):
        if isinstance(obj, str):
            return clean_text(obj)
        elif isinstance(obj, list):
            return [clean_obj(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: clean_obj(v) for k, v in obj.items()}
        return obj

    enriched_post = clean_obj(enriched_post)

    with open(processed_file_path, "w", encoding="utf-8") as f:
        json.dump(enriched_post, f, ensure_ascii=False, indent=2)



def extract_metadata(post: str):
    post = clean_text(post)  
    template = f"""
    You are given a LinkedIn post. You need to extract number of lines, language of the post and tags.
    1. Return a valid JSON. No preamble. 
    2. JSON object should have exactly three keys: line_count, language and tags. 
    3. tags is an array of text tags. Extract maximum two tags.
    4. Language should be English or Hinglish (Hinglish means hindi + english)
    
    Here is the actual post on which you need to perform this task:  
    {post}
    """
    response = llm.invoke(template)

    try:
        json_parser = JsonOutputParser()
        output = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Unable to parse LLM response into JSON")

    return output


if __name__ == "__main__":
    process_post("data/raw_post.json", "data/processed_posts.json")
