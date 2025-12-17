# This file was created iteratatively with Gemini and Google Colab over a series of multiple prompts.
# Here are the prompts
"""
Prompt 1: given the QID of a Wikipedia article, I want a python script that uses a python 
library wrapper to query Wikidata. My interest is to find out whether the page belongs to 
a human entity and if so, what is their gender or sex, their profession, and their date of 
birth. Can you write such a script. 

Prompt 2: This was great and it worked really well. I want now to focus on non-human entities. 
What kind of information I can get from the Wikidata page about such entities? To give you more 
context, I have access to Wikipedia pageviews data and I want to be able to say: pages about 
movies have more traffic than pages about sport events, or pages about science concepts are 
more popular than pages about historical events. Can you help extract meaningful data from 
the Wikidata page about non-humans? Create a new function about that. Imagine that we first 
check if a page is human or not, than if not human, call this function about the properties 
of such a page. 

Prompt 3: this works, are there any other attributes that can be added in addition to instance_of, 
country, and start date? 

Prompt 4: is there a dynamic way to get information? for example, is there a scraper of Wikidata? 
That is, if I go to this page: https://www.wikidata.org/wiki/Q83285, my scraper gives me a JSON 
of all attribute:values in the page. How is the HTML page created? Maybe data is passed as a 
JSON and we can use it directly. 

Prompt 5: now that I have the Property IDs for an entity in Wikidata, how do I get the corresponding
values of what they mean and what their values are for a given QID.

Prompt 6: okay, you got the names of the property IDs in the entry for the QID I provided you, 
but now I want to see their values, for example, what is the value for the P17 (country) for this 
QID, and so on. You'll need a second function to do this, at the end, I would like pairs like: 
"country": "Albania", "population": 113,249, and so on.
"""


import requests
import json, os

WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"

def fetch_complete_entity_data(qid):
    """
    Fetches all available structured data for a single Wikidata entity (QID)
    using the official Wikibase API action=wbgetentities.

    Args:
        qid (str): The Wikidata Item ID (e.g., 'Q83285' for Durres).

    Returns:
        dict: The complete raw JSON data for the entity, or an error dictionary.
    """

    # Parameters for the MediaWiki API, using the 'wbgetentities' action
    params = {
        'action': 'wbgetentities',
        'ids': qid,
        'format': 'json',
        # Request all relevant data: claims (properties), labels, descriptions, sitelinks (Wikipedia links)
        'props': 'claims|labels|descriptions|sitelinks|aliases',
    }

    # Add a User-Agent header as recommended by Wikidata API policies
    # https://www.wikidata.org/wiki/Wikidata:Contact_the_development_team#User-Agent
    headers = {
        'User-Agent': 'Colab-Wikidata-Example/1.0 (https://colab.research.google.com; colab-user@example.com)'
    }

    try:
        response = requests.get(WIKIDATA_API_ENDPOINT, 
                                params=params, 
                                headers=headers, 
                                timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        data = response.json()

        # Check for potential errors in the API response structure
        if 'error' in data:
            return {"error": f"API Error for {qid}: {data['error']['info']}"}

        # The core data is nested under ['entities'][qid]
        entity_data = data.get('entities', {}).get(qid)

        if entity_data:
            return entity_data
        else:
            return {"error": f"Entity {qid} not found or no data returned."}

    except requests.exceptions.RequestException as e:
        return {"error": f"Network or API request error: {e}"}
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON response."}

def _chunk_list(lst, n):
    """
    Yields successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def fetch_labels_for_qids(qids: list[str], lang='en'):
    """
    Fetches labels for a list of Wikidata QIDs or Property IDs.
    Handles API limits by chunking the requests.

    Args:
        qids (list[str]): A list of Wikidata Item IDs or Property IDs (e.g., ['Q515', 'P31']).
        lang (str): The language code for the labels (default is 'en').

    Returns:
        dict: A dictionary mapping QID to its label, or an error dictionary.
    """
    if not qids:
        return {}

    # Wikidata API limit for 'ids' parameter is typically 50
    MAX_IDS_PER_REQUEST = 50
    all_labels_map = {}

    # Chunk the QID list to respect the API limit
    for qid_chunk in _chunk_list(qids, MAX_IDS_PER_REQUEST):
        params = {
            'action': 'wbgetentities',
            'ids': '|'.join(qid_chunk), # Join QIDs with '|' for multiple requests
            'format': 'json',
            'props': 'labels',
            'languages': lang,
        }

        headers = {
            'User-Agent': 'Colab-Wikidata-Example/1.0 (https://colab.research.google.com; colab-user@example.com)'
        }

        try:
            response = requests.get(WIKIDATA_API_ENDPOINT, params=params, headers=headers, timeout=10)
            response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)

            data = response.json()

            if 'error' in data:
                # If an error occurs in one chunk, return it immediately or log and continue
                return {"error": f"API Error fetching labels for chunk {qid_chunk}: {data['error']['info']}"}

            for qid_key, entity_info in data.get('entities', {}).items():
                label = entity_info.get('labels', {}).get(lang, {}).get('value')
                if label:
                    all_labels_map[qid_key] = label

        except requests.exceptions.RequestException as e:
            return {"error": f"Network or API request error for chunk {qid_chunk}: {e}"}
        except json.JSONDecodeError:
            return {"error": "Failed to decode JSON response for a label chunk."}

    return all_labels_map

def extract_labeled_claim_values(claims: dict, property_labels: dict) -> dict:
    """
    Extracts the main value for each claim, resolves QID values to labels,
    and returns a dictionary of 'property_label': 'value' pairs.

    Args:
        claims (dict): The 'claims' section of a Wikidata entity's data.
        property_labels (dict): A dictionary mapping Property IDs (P-numbers) to their labels.

    Returns:
        dict: A dictionary where keys are property labels and values are their extracted/resolved values.
    """
    labeled_values = {}
    qids_to_resolve = set() # Collect all QIDs that need labels

    # First pass: Extract raw values and collect QIDs
    extracted_raw_values = {}
    for prop_id, statements in claims.items():
        prop_label = property_labels.get(prop_id, prop_id) # Use ID if label not found
        
        # We often care about the primary value of the first statement for simplicity
        if statements:
            main_snak = statements[0].get('mainsnak')
            if not main_snak or 'datavalue' not in main_snak: # Skip if no main value
                continue

            data_value = main_snak['datavalue']
            value_type = data_value.get('type')

            if value_type == 'wikibase-entityid':
                qid_value = data_value['value']['id']
                extracted_raw_values[prop_label] = qid_value # Store QID for later resolution
                qids_to_resolve.add(qid_value)
            elif value_type == 'string' or value_type == 'external-id':
                extracted_raw_values[prop_label] = data_value['value']
            elif value_type == 'quantity':
                # Format quantity with unit if available
                amount = data_value['value']['amount']
                unit = data_value['value'].get('unit', '').replace('http://www.wikidata.org/entity/', '')
                if unit and unit != '1': # '1' is the URI for dimensionless unit
                    # Attempt to add unit to QID list for resolution
                    if unit.startswith('Q'):
                        qids_to_resolve.add(unit)
                        extracted_raw_values[prop_label] = (amount, unit) # Store as tuple for later unit resolution
                    else:
                        extracted_raw_values[prop_label] = f"{amount} {unit}" # Simple string for non-QID units
                else:
                    extracted_raw_values[prop_label] = amount
            elif value_type == 'time':
                # Simple representation for time
                extracted_raw_values[prop_label] = data_value['value']['time']
            elif value_type == 'globecoordinate':
                latitude = data_value['value']['latitude']
                longitude = data_value['value']['longitude']
                extracted_raw_values[prop_label] = f"Lat: {latitude}, Lon: {longitude}"
            elif value_type == 'monolingualtext':
                extracted_raw_values[prop_label] = data_value['value']['text']
            # Add more types as needed
            else:
                # For unhandled types or complex structures, just show the raw datavalue
                extracted_raw_values[prop_label] = f"[Unhandled Type: {value_type}]"

    # Second pass: Resolve QID values and units to labels
    if qids_to_resolve:
        resolved_value_labels = fetch_labels_for_qids(list(qids_to_resolve))
        if "error" in resolved_value_labels:
            print(f"Warning: Could not resolve some value labels: {resolved_value_labels['error']}")
            # Proceed with raw QIDs if resolution fails
            pass

        for prop_label, value in extracted_raw_values.items():
            if isinstance(value, str) and value.startswith('Q'):
                labeled_values[prop_label] = resolved_value_labels.get(value, value) # Use raw QID if label not found
            elif isinstance(value, tuple) and len(value) == 2 and value[1].startswith('Q'): # Handle quantity with QID unit
                amount, unit_qid = value
                unit_label = resolved_value_labels.get(unit_qid, unit_qid)
                labeled_values[prop_label] = f"{amount} {unit_label}"
            else:
                labeled_values[prop_label] = value
    else:
        labeled_values = extracted_raw_values # No QIDs to resolve

    return labeled_values

def test_one(QID):
    """
    Demonstrates fetching the complete JSON data for a given QID string
    and then resolving labels for properties and their values.
    """
    # The entity for the Durres city
    qid_example = QID
    print(f"--- Fetching ALL structured data for {qid_example} \n")

    entity_data = fetch_complete_entity_data(qid_example)

    if "error" in entity_data:
        print(f"Error: {entity_data['error']}")
        return

    # Display main entity's label and description
    print(f"--- Main Entity Details ({qid_example}) ---")
    entity_label = entity_data.get('labels', {}).get('en', {}).get('value', 'No label found')
    entity_description = entity_data.get('descriptions', {}).get('en', {}).get('value', 'No description found')
    print(f"Label: {entity_label}")
    print(f"Description: {entity_description}\n")

    print("--- Full Raw JSON Structure (Truncated for readability) ---")

    # We will print the Claims section specifically to show the attribute:value pairs
    claims = entity_data.get('claims', {}) # This is the full claims dict
    print(f"\nTotal Properties (Claims) Found: {len(claims)}\n")

    # Get labels for the property IDs themselves
    property_ids = list(claims.keys())
    property_labels = fetch_labels_for_qids(property_ids)
    if "error" in property_labels:
        #print(f"Error fetching property labels: {property_labels['error']}")
        property_labels = {pid: pid for pid in property_ids} # Fallback to IDs if labels fail
    else:
        #print("Property IDs found for this entity:")
        # Print property IDs with their labels
        labeled_properties_overview = {pid: property_labels.get(pid, 'Label Not Found') for pid in property_ids}
        #print(json.dumps(labeled_properties_overview, indent=2))

    # Now, extract and label the claim values
    print("\n--- Extracted Labeled Claim Values ---")
    labeled_claim_values = extract_labeled_claim_values(claims, property_labels)
    print(json.dumps(labeled_claim_values, indent=2, ensure_ascii=False))

    print("\n--- Details for 'P31' (instance of) ---")

    if 'P31' in claims:
        # P31 is 'instance of', and it will contain an array of statements
        p31_statements = claims['P31']

        # Iterate over the values found for P31
        extracted_value_qids = []
        for statement in p31_statements:
            # The value is usually nested deep in the datavalue section
            main_snak = statement['mainsnak']
            if main_snak['datavalue']['type'] == 'wikibase-entityid':
                value_qid = main_snak['datavalue']['value']['id']
                extracted_value_qids.append(value_qid)

        # Get the label for the P31 property itself
        p31_label = property_labels.get('P31', 'Label Not Found for P31')
        print(f"Property P31 label: '{p31_label}'")

        # Get the labels for the extracted QID values
        value_labels = fetch_labels_for_qids(extracted_value_qids)

        if "error" in value_labels:
            print(f"Error fetching value labels: {value_labels['error']}")
        else:
            print(f"Raw QID values for 'instance of' (P31): {extracted_value_qids}")
            labeled_values = [value_labels.get(qid, 'Label Not Found') for qid in extracted_value_qids]
            print(f"Labeled values for 'instance of' (P31): {labeled_values}")
    else:
        print("P31 property not found in claims.")

    print("\n------------------------------------------------------------")
    print("This raw data contains every single piece of structured information available for the entity.")


def process_qids_to_jsonl(qid_list, output_filename="entity_data.jsonl"):
    """
    Processes a list of QIDs, fetches structured data, labels it, and stores
    the results (or errors) into a JSONL file.
    
    Args:
        qid_list (list): A list of QID strings (e.g., ['Q534', 'Q142', 'Q999']).
        output_filename (str): The name of the JSONL file to write results to.
    """
    print(f"Starting processing for {len(qid_list)} QIDs.")
    print(f"Results will be written to '{output_filename}'.")
    
    successful_count = 0
    failed_count = 0

    with open(output_filename, 'w', encoding='utf-8') as f:
        for qid in qid_list:
            print(f"Processing {qid}...")
            
            # Initialize the base record structure
            record = {"QID": qid, "status": "failed", "error_message": None}
            
            try:
                # 1. Fetch raw entity data (using your existing function)
                entity_data = fetch_complete_entity_data(qid)

                if "error" in entity_data:
                    # Handle API/Not Found error directly
                    record["error_message"] = entity_data['error']
                    failed_count += 1
                else:
                    # 2. Extract basic details
                    entity_label = entity_data.get('labels', {}).get('en', {}).get('value', 'No label found')
                    entity_description = entity_data.get('descriptions', {}).get('en', {}).get('value', 'No description found')
                    claims = entity_data.get('claims', {})
                    
                    # 3. Get labels for the properties themselves (using your existing function)
                    property_ids = list(claims.keys())
                    property_labels = fetch_labels_for_qids(property_ids)

                    if "error" in property_labels:
                        # Fallback for label fetching error
                        property_labels = {pid: pid for pid in property_ids} 
                        print(f"  Warning: Failed to fetch property labels for {qid}. Using IDs.")
                    
                    # 4. Extract and label all claim values (using your existing function)
                    labeled_claim_values = extract_labeled_claim_values(claims, property_labels)

                    # 5. Structure the final dictionary for successful outcome
                    record.update({
                        "status": "success",
                        "label": entity_label,
                        "description": entity_description,
                        "attributes": labeled_claim_values
                    })
                    record.pop("error_message") # Remove error key on success
                    successful_count += 1
            
            except Exception as e:
                # Catch any unexpected execution errors
                record["error_message"] = f"Unexpected execution error: {type(e).__name__} - {e}"
                failed_count += 1

            # 6. Write the final record (whether success or failure) to the JSONL file
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print("\n--- Processing Complete ---")
    print(f"Total Processed: {len(qid_list)}")
    print(f"Successful Records: {successful_count}")
    print(f"Failed Records: {failed_count}")
    print("---------------------------\n")

if __name__ == "__main__":
    #test_one("Q83285") # Article about Durres
    #test_one("Q7186")  # Article about Marie Kurie

    # I'm putting the list here, but you'll have a file with a list of QIDs here.
    qid_list_to_process = [
        "Q83285",    
        "Q7186",     
        "Q777",     
        "Q999",     
    ]
    
    with open("top_10000_qids_d.txt", "r") as f:
        q_list = [line.strip() for line in f if line.strip()]

    output_file = "entity_results_after.jsonl"

    # Run the main function
    process_qids_to_jsonl(q_list, output_file)

