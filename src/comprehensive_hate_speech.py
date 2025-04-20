import torch
import torch.nn.functional as F
import re
import string
import os
from collections import defaultdict
from hate_speech_model import classify_text
from religious_hate_speech import ReligiousHateSpeechDetector

class ComprehensiveHateSpeechDetector:
    """A comprehensive detector that identifies multiple types of hate speech"""
    
    def __init__(self):
        # Initialize the religious hate speech detector
        self.religious_detector = ReligiousHateSpeechDetector()
        
        # Racial/ethnic groups and terms
        self.racial_ethnic_groups = {
            'black': ['black', 'african', 'negro', 'colored', 'african american', 'jamaican', 'haitian', 'n-word', 'nigger', 'nigga'],
            'asian': ['asian', 'chinese', 'japanese', 'korean', 'vietnamese', 'filipino', 'thai', 'indian', 'pakistani', 'bangladeshi'],
            'latino': ['latino', 'latina', 'hispanic', 'mexican', 'puerto rican', 'cuban', 'dominican', 'colombian', 'salvadoran', 'venezuelan'],
            'middle_eastern': ['arab', 'middle eastern', 'persian', 'turkish', 'lebanese', 'syrian', 'iraqi', 'iranian', 'egyptian', 'saudi', 'palestinian'],
            'indigenous': ['native american', 'indigenous', 'first nations', 'american indian', 'aboriginal', 'inuit', 'metis'],
            'white': ['white', 'caucasian', 'european', 'anglo', 'western']
        }
        
        # Racial/ethnic stereotypes
        self.racial_stereotypes = {
            'black': ['thug', 'criminal', 'ghetto', 'gang', 'welfare', 'lazy', 'violent', 'drug dealer', 'monkey', 'ape'],
            'asian': ['virus', 'dog eater', 'bat eater', 'smart', 'math', 'small eyes', 'tiger parent', 'kung flu', 'china virus', 'foreigner'],
            'latino': ['illegal', 'alien', 'border jumper', 'cartel', 'drug dealer', 'gang member', 'welfare', 'anchor baby', 'lazy'],
            'middle_eastern': ['terrorist', 'bomb', 'isis', 'camel', 'sand', 'desert', 'extremist', 'radical', 'backwards'],
            'indigenous': ['savage', 'alcoholic', 'casino', 'primitive', 'redskin', 'extinct', 'tribal', 'backward'],
            'white': ['privileged', 'racist', 'colonizer', 'cracker', 'white trash', 'hillbilly', 'inbred', 'mayo', 'karen']
        }
        
        # Gender-based groups
        self.gender_groups = {
            'women': ['woman', 'women', 'female', 'girl', 'girls', 'lady', 'ladies', 'chick', 'mama', 'mother'],
            'men': ['man', 'men', 'male', 'boy', 'boys', 'guy', 'dude', 'father', 'bro', 'brother']
        }
        
        # Gender stereotypes
        self.gender_stereotypes = {
            'women': ['kitchen', 'sandwich', 'dishwasher', 'emotional', 'hysterical', 'bossy', 'nag', 'gold digger', 'slut', 'whore', 'bitch'],
            'men': ['simp', 'soy boy', 'cuck', 'virgin', 'incel', 'weak', 'soft', 'beta']
        }
        
        # LGBTQ+ groups
        self.lgbtq_groups = {
            'gay': ['gay', 'homosexual', 'queer', 'homo', 'same sex'],
            'lesbian': ['lesbian', 'dyke', 'queer woman'],
            'bisexual': ['bisexual', 'bi'],
            'transgender': ['transgender', 'trans', 'transsexual', 'mtf', 'ftm'],
            'nonbinary': ['non-binary', 'nonbinary', 'enby', 'genderqueer', 'genderfluid']
        }
        
        # LGBTQ+ stereotypes
        self.lgbtq_stereotypes = {
            'gay': ['fag', 'faggot', 'fairy', 'sissy', 'groomer', 'pedophile', 'unnatural', 'sin', 'disease', 'aids'],
            'lesbian': ['dyke', 'butch', 'man-hater', 'confused', 'just needs a man'],
            'bisexual': ['confused', 'greedy', 'promiscuous', 'unfaithful', 'attention-seeking'],
            'transgender': ['tranny', 'trap', 'shemale', 'it', 'confused', 'mental illness', 'mutilated', 'delusional', 'groomer'],
            'nonbinary': ['confused', 'attention-seeking', 'made up', 'not real', 'snowflake', 'special snowflake']
        }
        
        # Sexual/explicit terms and offensive language - NEW CATEGORY
        self.sexual_explicit_terms = [
            'fuck', 'fucking', 'fucked', 'fucker', 'motherfucker', 
            'pussy', 'cunt', 'dick', 'cock', 'penis', 'vagina', 
            'ass', 'asshole', 'shit', 'bullshit', 'crap',
            'tits', 'boobs', 'breasts', 'cum', 'jizz', 'semen',
            'sex', 'sexual', 'sexy', 'horny', 'masturbate',
            'oral', 'anal', 'bitch', 'bastard', 'damn',
            # Additional terms and compound anatomical terms
            'big boobs', 'small boobs', 'huge boobs', 'nice boobs',
            'big tits', 'small tits', 'huge tits', 'nice tits',
            'wet pussy', 'tight pussy', 'big ass', 'fat ass', 'nice ass',
            'tight ass', 'big butt', 'large breasts', 'small breasts',
            'nipples', 'butthole', 'buttocks', 'rear end', 'private parts',
            'genitals', 'genital', 'genital area', 'crotch',
            'blow job', 'blowjob', 'handjob', 'hand job', 'rimjob', 'rim job',
            'jerking off', 'jerk off', 'wanking', 'wank', 'fingering',
            'titjob', 'tit job', 'titty fuck', 'titfuck'
        ]
        
        # Anatomical term patterns for regex matching (handles variations better)
        self.anatomical_patterns = [
            r'big\s+(?:boobs|tits|breasts|ass|butt)',
            r'(?:nice|huge|large|small|tight|wet|hot)\s+(?:boobs|tits|breasts|ass|butt|pussy)',
            r'(?:suck|lick|eat|finger)\s+(?:boobs|tits|breasts|ass|butt|pussy)',
            r'(?:show|see|nice|sexy)\s+(?:boobs|tits|breasts|ass|butt|pussy)'
        ]
        
        # Disability groups
        self.disability_groups = {
            'physical': ['disabled', 'wheelchair', 'cripple', 'handicapped', 'paralyzed', 'amputee'],
            'mental': ['mentally ill', 'mental illness', 'psychiatric', 'crazy', 'schizo', 'bipolar', 'depression'],
            'developmental': ['autistic', 'autism', 'down syndrome', 'intellectual disability', 'special needs'],
            'sensory': ['blind', 'deaf', 'mute', 'hard of hearing']
        }
        
        # Disability stereotypes
        self.disability_stereotypes = {
            'physical': ['burden', 'useless', 'helpless', 'dependent', 'incapable', 'inspiration porn', 'faker'],
            'mental': ['crazy', 'psycho', 'dangerous', 'unstable', 'attention-seeking', 'drama', 'faker'],
            'developmental': ['retard', 'retarded', 'stupid', 'dumb', 'idiot', 'slow', 'childish', 'burden'],
            'sensory': ['helpless', 'useless', 'burden', 'faking', 'special treatment']
        }
        
        # Nationality/immigrant groups
        self.nationality_groups = {
            'immigrant': ['immigrant', 'migrant', 'foreigner', 'alien', 'refugee', 'asylum seeker', 'non-citizen'],
            'nationality': ['american', 'british', 'canadian', 'australian', 'french', 'german', 'mexican', 'chinese', 'indian', 'russian']
        }
        
        # Nationality/immigrant stereotypes
        self.nationality_stereotypes = {
            'immigrant': ['illegal', 'criminal', 'job stealer', 'freeloader', 'welfare', 'invasion', 'disease', 'parasite', 'anchor baby'],
            'nationality': ['invader', 'not welcome', 'go back', 'leave', 'deportation', 'ban', 'wall']
        }
        
        # Generalizations, negative attributes and hate actions (shared across categories)
        self.generalizations = [
            'all', 'every', 'they all', 'they are all', 'all of them', 'those people',
            'these people', 'that kind', 'their kind', 'always', 'never',
            'as usual', 'typical', 'none of them', 'all of them'
        ]
        
        self.negative_attributes = [
            'dangerous', 'evil', 'bad', 'threat', 'violent', 'hateful',
            'radical', 'extremist', 'criminal', 'immoral', 'dirty', 'disgusting',
            'primitive', 'backwards', 'uncivilized', 'should be banned', 'should not be allowed',
            'don\'t belong', 'go back', 'kick them out', 'not welcome', 'shouldn\'t be trusted',
            'can\'t be trusted', 'are lying', 'are deceptive', 'brainwashed', 'stupid',
            'subhuman', 'animals', 'beasts', 'vermin', 'pests', 'infestation', 'plague'
        ]
        
        self.hate_actions = [
            'ban', 'deport', 'kill', 'attack', 'eliminate', 'remove', 'get rid of',
            'throw out', 'shouldn\'t be allowed', 'don\'t belong', 'need to leave',
            'exterminate', 'genocide', 'beat up', 'lock up', 'imprison', 'exclude',
            'segregate', 'separate', 'isolate', 'quarantine', 'eradicate', 'hang',
            'shoot', 'assault', 'harass', 'punch', 'slap', 'die', 'death to'
        ]
    
    def detect_hate_speech(self, text):
        """
        Detect various types of hate speech in text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Analysis results
        """
        text_lower = text.lower()
        
        # Initialize results
        results = {
            'is_hate_speech': False,
            'hate_category': None,
            'targeted_group': None,
            'hate_score': 0.0,
            'patterns_found': []
        }
        
        # First check religious hate speech
        religious_results = self.religious_detector.detect(text)
        if religious_results['is_religious_hate_speech']:
            results['is_hate_speech'] = True
            results['hate_category'] = 'religious'
            results['targeted_group'] = religious_results['targeted_religion']
            results['hate_score'] = religious_results['hate_score']
            results['patterns_found'] = religious_results['patterns_found']
            return results
        
        # Check racial/ethnic hate speech
        targeted_races = []
        for race, terms in self.racial_ethnic_groups.items():
            for term in terms:
                if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                    targeted_races.append(race)
                    results['patterns_found'].append(f'Racial/ethnic reference: {term} ({race})')
                    break
        
        # Check racial stereotypes
        for race in targeted_races:
            if race in self.racial_stereotypes:
                for stereotype in self.racial_stereotypes[race]:
                    if re.search(r'\b' + re.escape(stereotype) + r'\b', text_lower):
                        results['patterns_found'].append(f'Racial stereotype: {stereotype} ({race})')
                        results['hate_score'] += 0.4
                        break
        
        # If we found racial/ethnic references and stereotypes
        if targeted_races and results['hate_score'] > 0:
            results['hate_category'] = 'racial/ethnic'
            results['targeted_group'] = targeted_races[0] if len(targeted_races) == 1 else targeted_races
        
        # Check gender-based hate speech
        if not results['is_hate_speech']:
            targeted_genders = []
            for gender, terms in self.gender_groups.items():
                for term in terms:
                    if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                        targeted_genders.append(gender)
                        results['patterns_found'].append(f'Gender reference: {term} ({gender})')
                        break
            
            # Check gender stereotypes
            for gender in targeted_genders:
                if gender in self.gender_stereotypes:
                    for stereotype in self.gender_stereotypes[gender]:
                        if re.search(r'\b' + re.escape(stereotype) + r'\b', text_lower):
                            results['patterns_found'].append(f'Gender stereotype: {stereotype} ({gender})')
                            results['hate_score'] += 0.4
                            break
            
            # If we found gender references and stereotypes
            if targeted_genders and results['hate_score'] > 0:
                results['hate_category'] = 'gender'
                results['targeted_group'] = targeted_genders[0] if len(targeted_genders) == 1 else targeted_genders
        
        # Check LGBTQ+ hate speech
        if not results['is_hate_speech']:
            targeted_lgbtq = []
            for identity, terms in self.lgbtq_groups.items():
                for term in terms:
                    if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                        targeted_lgbtq.append(identity)
                        results['patterns_found'].append(f'LGBTQ+ reference: {term} ({identity})')
                        break
            
            # Check LGBTQ+ stereotypes
            for identity in targeted_lgbtq:
                if identity in self.lgbtq_stereotypes:
                    for stereotype in self.lgbtq_stereotypes[identity]:
                        if re.search(r'\b' + re.escape(stereotype) + r'\b', text_lower):
                            results['patterns_found'].append(f'LGBTQ+ stereotype: {stereotype} ({identity})')
                            results['hate_score'] += 0.4
                            break
            
            # If we found LGBTQ+ references and stereotypes
            if targeted_lgbtq and results['hate_score'] > 0:
                results['hate_category'] = 'lgbtq+'
                results['targeted_group'] = targeted_lgbtq[0] if len(targeted_lgbtq) == 1 else targeted_lgbtq
        
        # Check disability hate speech
        if not results['is_hate_speech']:
            targeted_disabilities = []
            for disability, terms in self.disability_groups.items():
                for term in terms:
                    if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                        targeted_disabilities.append(disability)
                        results['patterns_found'].append(f'Disability reference: {term} ({disability})')
                        break
            
            # Check disability stereotypes
            for disability in targeted_disabilities:
                if disability in self.disability_stereotypes:
                    for stereotype in self.disability_stereotypes[disability]:
                        if re.search(r'\b' + re.escape(stereotype) + r'\b', text_lower):
                            results['patterns_found'].append(f'Disability stereotype: {stereotype} ({disability})')
                            results['hate_score'] += 0.4
                            break
            
            # If we found disability references and stereotypes
            if targeted_disabilities and results['hate_score'] > 0:
                results['hate_category'] = 'disability'
                results['targeted_group'] = targeted_disabilities[0] if len(targeted_disabilities) == 1 else targeted_disabilities
        
        # Check nationality/immigrant hate speech
        if not results['is_hate_speech']:
            targeted_nationalities = []
            for category, terms in self.nationality_groups.items():
                for term in terms:
                    if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                        targeted_nationalities.append(category)
                        results['patterns_found'].append(f'Nationality/immigrant reference: {term} ({category})')
                        break
            
            # Check nationality/immigrant stereotypes
            for category in targeted_nationalities:
                if category in self.nationality_stereotypes:
                    for stereotype in self.nationality_stereotypes[category]:
                        if re.search(r'\b' + re.escape(stereotype) + r'\b', text_lower):
                            results['patterns_found'].append(f'Nationality/immigrant stereotype: {stereotype} ({category})')
                            results['hate_score'] += 0.4
                            break
            
            # If we found nationality/immigrant references and stereotypes
            if targeted_nationalities and results['hate_score'] > 0:
                results['hate_category'] = 'nationality/immigrant'
                results['targeted_group'] = targeted_nationalities[0] if len(targeted_nationalities) == 1 else targeted_nationalities
        
        # Check sexual/explicit terms and offensive language
        if not results['is_hate_speech']:
            for term in self.sexual_explicit_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                    results['patterns_found'].append(f'Sexual/explicit term: {term}')
                    results['hate_score'] += 0.5
                    results['hate_category'] = 'sexual/explicit'
                    results['targeted_group'] = 'general'
                    break
            
            # Check anatomical term patterns
            for pattern in self.anatomical_patterns:
                if re.search(pattern, text_lower):
                    matches = re.findall(pattern, text_lower)
                    match_text = matches[0] if matches else pattern
                    results['patterns_found'].append(f'Anatomical pattern match: {match_text}')
                    results['hate_score'] += 0.5
                    results['hate_category'] = 'sexual/explicit'
                    results['targeted_group'] = 'general'
                    break
        
        # Check for generalizations (common to all categories)
        for gen in self.generalizations:
            if re.search(r'\b' + re.escape(gen) + r'\b', text_lower):
                results['patterns_found'].append(f'Generalization: {gen}')
                results['hate_score'] += 0.3
                break
        
        # Check for negative attributes (common to all categories)
        for attr in self.negative_attributes:
            if re.search(r'\b' + re.escape(attr) + r'\b', text_lower):
                results['patterns_found'].append(f'Negative attribute: {attr}')
                results['hate_score'] += 0.3
                break
        
        # Check for hate actions (common to all categories)
        for action in self.hate_actions:
            if re.search(r'\b' + re.escape(action) + r'\b', text_lower):
                results['patterns_found'].append(f'Hate action: {action}')
                results['hate_score'] += 0.4
                break
        
        # If hate score is significant, classify as hate speech
        if results['hate_score'] > 0.3:
            results['is_hate_speech'] = True
            
            # Cap the hate score at 1.0
            results['hate_score'] = min(results['hate_score'], 1.0)
        
        return results

# For integration with model
def apply_comprehensive_hate_detection(text, model_result):
    """
    Apply comprehensive hate speech detection to model results
    
    Args:
        text (str): Original text
        model_result (dict): Result from classify_text function
        
    Returns:
        dict: Updated model result with comprehensive hate speech detection
    """
    # Create detector
    detector = ComprehensiveHateSpeechDetector()
    
    # Get detection results
    detection_result = detector.detect_hate_speech(text)
    
    # If hate speech is detected
    if detection_result['is_hate_speech']:
        # Only override if the model classified as non-hate speech with high confidence
        if model_result['prediction'] == 'Non-Hate Speech' and model_result['confidence'] > 0.65:
            # Update prediction
            model_result['prediction'] = 'Hate Speech'
            model_result['prediction_code'] = 1
            
            # Set confidence directly based on hate score without minimum threshold
            # This allows confidence to reflect actual intensity of hate speech detected
            model_result['confidence'] = detection_result['hate_score']
            
            # Add hate speech information
            model_result['comprehensive_hate_detection'] = detection_result
            
            # Safely handle the category (might be None in some cases)
            if detection_result['hate_category'] is not None:
                category_name = detection_result['hate_category'].capitalize()
            else:
                category_name = "Unspecified"
                
            model_result['override_reason'] = f"{category_name} hate speech detected"
    
    return model_result

# Updated classification function with comprehensive hate speech detection
def classify_text_with_comprehensive_detection(model, vocab, text, use_advanced_preprocessing=True):
    """Enhanced classify_text function with comprehensive hate speech detection"""
    # Get base classification
    result = classify_text(model, vocab, text, use_advanced_preprocessing)
    
    # Apply comprehensive hate speech detection
    enhanced_result = apply_comprehensive_hate_detection(text, result)
    
    return enhanced_result

def get_category_probabilities(text):
    """
    Get probability scores for different hate speech categories for a given text
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: Dictionary with category names as keys and probability scores as values
    """
    # Create detector
    detector = ComprehensiveHateSpeechDetector()
    
    # Get detection results
    detection_result = detector.detect_hate_speech(text)
    
    # Base probabilities for each category
    probabilities = {
        'Religious': 0.0,
        'Racial/Ethnic': 0.0, 
        'Gender-based': 0.0,
        'LGBTQ+': 0.0,
        'Disability-related': 0.0,
        'Nationality/Immigrant': 0.0,
        'Sexual/Explicit': 0.0  # Added new category
    }
    
    # If hate speech is detected, assign probabilities based on category
    if detection_result['is_hate_speech']:
        # Get the detected category (which could be None)
        category = detection_result['hate_category']
        
        # Map the detected category to our predefined categories
        if category == 'religious':
            probabilities['Religious'] = detection_result['hate_score']
        elif category == 'racial/ethnic':
            probabilities['Racial/Ethnic'] = detection_result['hate_score']
        elif category == 'gender':
            probabilities['Gender-based'] = detection_result['hate_score']
        elif category == 'lgbtq+':
            probabilities['LGBTQ+'] = detection_result['hate_score']
        elif category == 'disability':
            probabilities['Disability-related'] = detection_result['hate_score']
        elif category == 'nationality/immigrant':
            probabilities['Nationality/Immigrant'] = detection_result['hate_score']
        elif category == 'sexual/explicit':  # Added new category
            probabilities['Sexual/Explicit'] = detection_result['hate_score']
        elif category is None:
            # If category is None but hate speech was detected,
            # distribute a small probability across all categories
            for key in probabilities:
                probabilities[key] = detection_result['hate_score'] * 0.2
            
        # For non-matched categories, provide a small baseline probability
        # based on the general hate detection to indicate potential overlap
        baseline = detection_result['hate_score'] * 0.1
        for key in probabilities:
            if probabilities[key] == 0.0:
                probabilities[key] = baseline
    
    # If patterns were found but not classified as hate speech,
    # assign low probabilities based on the patterns
    elif detection_result['patterns_found']:
        # Calculate a base score from the hate score
        base_score = detection_result['hate_score'] * 0.5
        
        # Look for category indicators in the patterns
        for pattern in detection_result['patterns_found']:
            if 'religious' in pattern.lower():
                probabilities['Religious'] = max(probabilities['Religious'], base_score)
            elif 'racial' in pattern.lower() or 'ethnic' in pattern.lower():
                probabilities['Racial/Ethnic'] = max(probabilities['Racial/Ethnic'], base_score)
            elif 'gender' in pattern.lower():
                probabilities['Gender-based'] = max(probabilities['Gender-based'], base_score)
            elif 'lgbtq' in pattern.lower() or 'gay' in pattern.lower() or 'lesbian' in pattern.lower():
                probabilities['LGBTQ+'] = max(probabilities['LGBTQ+'], base_score)
            elif 'disability' in pattern.lower():
                probabilities['Disability-related'] = max(probabilities['Disability-related'], base_score)
            elif 'nationality' in pattern.lower() or 'immigrant' in pattern.lower():
                probabilities['Nationality/Immigrant'] = max(probabilities['Nationality/Immigrant'], base_score)
            elif 'sexual' in pattern.lower() or 'explicit' in pattern.lower():  # Added new category
                probabilities['Sexual/Explicit'] = max(probabilities['Sexual/Explicit'], base_score)
    
    return probabilities