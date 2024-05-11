import pandas
import os 


corporate_participants_delimiter = """================================================================================
Corporate Participants
================================================================================"""

conference_call_participants_delimiter = """================================================================================
Conference Call Participiants
================================================================================"""

presentation_delimiter = """================================================================================
Presentation"""

qna_delimiter = """================================================================================
Questions and Answers
"""

definitions_delimiter = """--------------------------------------------------------------------------------
Definitions
--------------------------------------------------------------------------------"""

delimiter = "--------------------------------------------------------------------------------"


def process_participants(corporate_participants):
    ''' Return lists with the corporate participants and their respective roles and companies'''
    corporate_participants = corporate_participants.split("\n")
    i = 0
    output = {}
    while i < len(corporate_participants):
        line = corporate_participants[i]
        if line.startswith(" *"):
            name = line[2:].strip()
            company, role = corporate_participants[i+1].split(" - ")
            company, role = company.strip(), role.strip()
            output[name] = (company, role)
            i += 2
        else:
            i += 1
    return output
            


def processing(data, path):
    corporate_participants = data.split(corporate_participants_delimiter)[1].split(conference_call_participants_delimiter)[0]
    conference_participants = data.split(conference_call_participants_delimiter)[1].split(presentation_delimiter)[0]

    # extract lines startign with "*"
    dic_corporate_participants = process_participants(corporate_participants)
    dic_conference_participants = process_participants(conference_participants)
    list_corporate_participants = list(dic_corporate_participants.keys())
    list_conference_participants = list(dic_conference_participants.keys())

    presentation = data.split(presentation_delimiter)[1].split(qna_delimiter)[0]
    qna = data.split(qna_delimiter)[1].split(definitions_delimiter)[0]

    list_text = []
    list_speaker = []
    type = []
    speaker_type = []
    speaker_company = []
    speaker_role = []
    # process every speaker line in the presentation
    presentation_lines = presentation.split(delimiter)
    for i in range(len(presentation_lines)-1):
        if i % 2 == 1:
            speaker = presentation_lines[i]
            text = presentation_lines[i+1]
            if "Operator" in speaker:
                speaker = "Operator"
                speaker_type.append("Operator")
                speaker_company.append("")
                speaker_role.append("")
            else:
                speaker = speaker.split(",")[0].split(";")[0].strip()
                if speaker in list_corporate_participants:
                    speaker_type.append("Corporate Participant")
                    speaker_company.append(dic_corporate_participants[speaker][0])
                    speaker_role.append(dic_corporate_participants[speaker][1])
                elif speaker in list_conference_participants:
                    speaker_type.append("Conference Participant")
                    speaker_company.append(dic_conference_participants[speaker][0])
                    speaker_role.append(dic_conference_participants[speaker][1])
            list_speaker.append(speaker)
            list_text.append(text.strip())
            type.append("presentation")

    # process every speaker line in the Q&A
    qna_lines = qna.split(delimiter)
    for i in range(len(qna_lines)-1):
        if i % 2 == 1:
            speaker = qna_lines[i]
            text = qna_lines[i+1]
            if "Operator" in speaker:
                speaker = "Operator"
                speaker_type.append("Operator")
                speaker_company.append("")
                speaker_role.append("")
            else:
                speaker = speaker.split(",")[0].split(";")[0].strip()
                if speaker in list_corporate_participants:
                    speaker_type.append("Corporate Participant")
                    speaker_company.append(dic_corporate_participants[speaker][0])
                    speaker_role.append(dic_corporate_participants[speaker][1])
                elif speaker in list_conference_participants:
                    speaker_type.append("Conference Participant")
                    speaker_company.append(dic_conference_participants[speaker][0])
                    speaker_role.append(dic_conference_participants[speaker][1])
            list_speaker.append(speaker)
            list_text.append(text.strip())
            type.append("qna")

    column_company_name = os.path.basename(path).split("-")[3]
    column_date = os.path.basename(path).split("-")[:3]
    column_date = "-".join(column_date)

    column_company_name = [column_company_name] * len(list_speaker)
    column_date = [column_date] * len(list_speaker)

    # create dataframe
    try:
        df = pandas.DataFrame({"speaker": list_speaker, "text": list_text, "type": type, "speaker_type": speaker_type, "speaker_company": speaker_company, "speaker_role": speaker_role, "company_name": column_company_name, "date": column_date})
    except:
        return None
    return df


def main(directory):
    path_list = []
    # all .txt files in sub directories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                path_list.append(os.path.join(root, file))
    dataframes = []
    for path in path_list:
        with open(path, 'r') as file:
            data = file.read()
        df = processing(data, path)
        if df is not None:
            dataframes.append(df)
    df = pandas.concat(dataframes)
    return df
