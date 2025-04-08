DBPEDIA = {
    # Zero-shot prompt to classify articles and return their category name.
    "ZERO_SHOT" : """You are an expert at classifying articles into the following categories:

    CATEGORIES:
    0. Company
    1. EducationalInstitution
    2. Artist
    3. Athlete
    4. OfficeHolder
    5. MeanOfTransportation
    6. Building
    7. NaturalPlace
    8. Village
    9. Animal
    10. Plant
    11. Album
    12. Film
    13. WrittenWork

    Read the following article then answer with the name of the category which suits it best.
    Answer with ONLY the name of the category, i.e. "Company"."""
}

OSHA = {
    "ZERO_SHOT" : """You are an expert at classifying OSHA injury reports into the following categories:

    CATEGORIES:
    0. Amputations
    1. Crushing injuries
    2. Cuts, lacerations
    3. Fractures
    4. Heat (thermal) burns, unspecified
    5. Internal injuries to organs and blood vessels of the trunk
    6. Intracranial injuries, unspecified
    7. Puncture wounds, except gunshot wounds
    8. Soreness, pain, hurt-nonspecified injury
    9. Traumatic injuries and disorders, unspecified

    Read the following OSHA injury report then answer with the name of the category which suits it best.
    Answer with ONLY the name of the category, i.e. "Fractures"."""
}
