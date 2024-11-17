import argparse
import pandas as pd
import time
from googlereviews import CSVProcessor
from PII.pii import TextAnalyzerService
from profanity_masker.main import profanity_masker
from sentiment_classifier.main import TextClassifier

def process_reviews(input_csv_path):
    start_time = time.perf_counter()
    
    # Step 1: Initial CSV Processing
    print(f"Processing input file: {input_csv_path}")
    columns_to_drop = ['business_name', 'author_name', 'photo', 'rating_category']
    
    # Read the input CSV file
    df = pd.read_csv(input_csv_path)
    
    # Step 2: Anonymization
    print("Performing text anonymization...")
    text_analyzer_service_model1 = TextAnalyzerService(model_choice="obi/deid_roberta_i2b2")
    anonymized_texts = []
    
    for index, row in df.iterrows():
        text = row.iloc[0]  # Assuming the first column contains the text
        entities_model1 = text_analyzer_service_model1.analyze_text(text)
        anonymized_text, req_dict = text_analyzer_service_model1.anonymize_text(
            text, 
            entities_model1, 
            operator="encrypt"
        )
        anonymized_texts.append(anonymized_text.text)
    
    df['Anonymized_Text'] = anonymized_texts
    df.to_csv("output_anonymized.csv", index=False)
    
    # Step 3: Profanity Masking
    print("Masking profanity...")
    masker = profanity_masker()
    df['Masked_Text'] = df['Anonymized_Text'].apply(lambda x: masker.mask_words(x))
    df.to_csv("output_masked.csv", index=False)
    
    # Step 4: Sentiment Classification
    print("Performing sentiment classification...")
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    classifier = TextClassifier(checkpoint)
    df['Classification_Result'] = df.iloc[:, 0].apply(lambda x: classifier.infer(x))
    
    # Calculate and print processing time
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    
    # Save final output
    output_path = "output_classified.csv"
    df.to_csv(output_path, index=False)
    print(f"Processing complete. Final output saved to: {output_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process review data with anonymization, profanity masking, and sentiment classification.')
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file containing reviews')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the reviews
    process_reviews(args.input_csv)

if __name__ == "__main__":
    main()
