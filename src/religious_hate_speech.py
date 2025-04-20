import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import string
import pickle
import os
import numpy as np
import json
from collections import defaultdict
from hate_speech_model import classify_text  # Import the classify_text function from hate_speech_model

class ReligiousHateSpeechDetector:
    """A specialized detector for religious hate speech"""
    
    def __init__(self):
        # Religious groups and terms
        self.religious_groups = {
            'islam': ['muslim', 'islam', 'islamic', 'quran', 'mosque', 'allah', 'ramadan', 'eid', 'muhammad'],
            'christianity': ['christian', 'christianity', 'bible', 'jesus', 'christ', 'church', 'catholic', 'protestant', 'evangelical'],
            'judaism': ['jew', 'jewish', 'judaism', 'rabbi', 'synagogue', 'torah', 'kosher', 'zionist', 'israel'],
            'hinduism': ['hindu', 'hinduism', 'temple', 'brahmin', 'karma', 'yoga', 'vedas', 'sanskrit', 'dharma'],
            'buddhism': ['buddhist', 'buddhism', 'buddha', 'dharma', 'nirvana', 'meditation', 'monk', 'temple'],
            'sikhism': ['sikh', 'sikhism', 'gurdwara', 'guru', 'punjabi', 'turban', 'khalsa'],
            'atheism': ['atheist', 'atheism', 'godless', 'secular', 'nonbeliever']
        }
        
        # Negative stereotype patterns by religion
        self.religious_stereotypes = {
            'islam': ['terrorist', 'bomb', 'violent', 'radical', 'extremist', 'jihad', 'isis', 'backwards'],
            'christianity': ['hypocrite', 'judgmental', 'bigot', 'fanatic', 'cult', 'brainwashed'],
            'judaism': ['greedy', 'conspiracy', 'control', 'money', 'cheap', 'globalist', 'zog'],
            'hinduism': ['cow', 'primitive', 'superstition', 'caste', 'dirty', 'smelly', 'curry'],
            'buddhism': ['cult', 'brainwashed', 'communist', 'china'],
            'sikhism': ['terrorist', 'taliban', 'muslim', 'arab'], # common mistaken stereotypes
            'atheism': ['immoral', 'communist', 'satanist', 'evil', 'sinner']
        }
        
        # Generalization phrases
        self.generalizations = [
            'all', 'every', 'they all', 'they are all', 'all of them', 'those people',
            'these people', 'that kind', 'their kind', 'always', 'never',
            'as usual', 'typical', 'none of them', 'all of them'
        ]
        
        # Negative attributes commonly used in hate speech
        self.negative_attributes = [
            'dangerous', 'evil', 'bad', 'threat', 'terrorist', 'violent', 'hateful',
            'radical', 'extremist', 'criminal', 'immoral', 'dirty', 'disgusting',
            'primitive', 'backwards', 'uncivilized', 'should be banned', 'should not be allowed',
            'don\'t belong', 'go back', 'kick them out', 'not welcome', 'shouldn\'t be trusted',
            'can\'t be trusted', 'are lying', 'are deceptive', 'brainwashed', 'stupid'
        ]
        
        # Actions often advocated in hate speech
        self.hate_actions = [
            'ban', 'deport', 'kill', 'attack', 'eliminate', 'remove', 'get rid of',
            'throw out', 'shouldn\'t be allowed', 'don\'t belong', 'need to leave'
        ]
    
    def detect(self, text):
        """
        Detect religious hate speech in text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Analysis results
        """
        text_lower = text.lower()
        results = {
            'is_religious_hate_speech': False,
            'targeted_religion': None,
            'hate_score': 0.0,
            'patterns_found': []
        }
        
        # Look for religious group mentions
        targeted_religions = []
        for religion, terms in self.religious_groups.items():
            for term in terms:
                if f' {term}' in f' {text_lower} ' or f'-{term}' in text_lower or f'{term}-' in text_lower:
                    targeted_religions.append(religion)
                    results['patterns_found'].append(f'Religious reference: {term} ({religion})')
                    break
        
        if not targeted_religions:
            return results
        
        # Look for generalizations
        has_generalization = False
        for gen in self.generalizations:
            if f' {gen} ' in f' {text_lower} ':
                has_generalization = True
                results['patterns_found'].append(f'Generalization: {gen}')
                results['hate_score'] += 0.3
                break
        
        # Look for negative attributes
        has_negative_attribute = False
        for attr in self.negative_attributes:
            if f' {attr} ' in f' {text_lower} ' or f' {attr}.' in f' {text_lower} ' or f' {attr},' in f' {text_lower} ':
                has_negative_attribute = True
                results['patterns_found'].append(f'Negative attribute: {attr}')
                results['hate_score'] += 0.3
                break
        
        # Look for hate actions
        has_hate_action = False
        for action in self.hate_actions:
            if f' {action} ' in f' {text_lower} ' or f' {action}.' in f' {text_lower} ' or f' {action},' in f' {text_lower} ':
                has_hate_action = True
                results['patterns_found'].append(f'Hate action: {action}')
                results['hate_score'] += 0.4
                break
        
        # Look for specific stereotypes for the targeted religions
        for religion in targeted_religions:
            if religion in self.religious_stereotypes:
                for stereotype in self.religious_stereotypes[religion]:
                    if f' {stereotype} ' in f' {text_lower} ' or f' {stereotype}.' in f' {text_lower} ' or f' {stereotype},' in f' {text_lower} ':
                        results['patterns_found'].append(f'Religious stereotype: {stereotype} ({religion})')
                        results['hate_score'] += 0.4
                        break
        
        # If has religious reference plus either generalization, negative attribute, or stereotype
        if has_generalization or has_negative_attribute or has_hate_action or results['hate_score'] > 0.3:
            results['is_religious_hate_speech'] = True
            results['targeted_religion'] = targeted_religions[0] if len(targeted_religions) == 1 else targeted_religions
            
            # Cap the hate score at 1.0
            results['hate_score'] = min(results['hate_score'], 1.0)
        
        return results

# For integration with model
def apply_religious_hate_detection(text, model_result):
    """Apply religious hate speech detection to model results
    
    Args:
        text (str): Original text
        model_result (dict): Result from classify_text function
        
    Returns:
        dict: Updated model result with religious hate speech detection
    """
    # Create detector
    detector = ReligiousHateSpeechDetector()
    
    # Get detection results
    detection_result = detector.detect(text)
    
    # If religious hate speech is detected
    if detection_result['is_religious_hate_speech']:
        # Only override if the model classified as non-hate speech with high confidence
        if model_result['prediction'] == 'Non-Hate Speech' and model_result['confidence'] > 0.7:
            # Update prediction
            model_result['prediction'] = 'Hate Speech'
            model_result['prediction_code'] = 1
            
            # Adjust confidence based on hate score
            model_result['confidence'] = max(detection_result['hate_score'], 0.7)
            
            # Add religious hate speech information
            model_result['religious_hate_speech'] = detection_result
            model_result['override_reason'] = 'Religious hate speech detected'
    
    return model_result

# Update classify_text function to use religious hate speech detection
def classify_text_with_religious_detection(model, vocab, text, use_advanced_preprocessing=True):
    """Enhanced classify_text function with religious hate speech detection"""
    # Get base classification
    result = classify_text(model, vocab, text, use_advanced_preprocessing)
    
    # Apply religious hate speech detection
    enhanced_result = apply_religious_hate_detection(text, result)
    
    return enhanced_result