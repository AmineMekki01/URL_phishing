
""" 
Module : constants.py
Description : This module contains the constants used in the application.
"""
feature_cols = [
            'url_length', 'num_dots', 'num_hyphens', 'num_digits', 'num_alpha', 
            'subdomain_level', 'path_level', 'count_dashes_hostname', 'no_https', 
            'random_string', 'ip_address', 'domain_in_subdomains', 'domain_in_paths', 
            'https_in_hostname', 'hostname_length', 'count_sensitive_words', 
            'url_entropy', 'num_@', 'num_?', 'num_-', 'num_=', 'num_dot', 
            'num_hash', 'num_percent', 'num_plus', 'num_dollar', 'num_exclamation', 
            'num_asterisk', 'num_comma', 'num_//', 'num_query_symbol', 
            'num_dns_answers', 'num_ip_octets', 'sum_ip_octets', 'has_ip_address', 
            'domain_length'
        ]

models_path = "./artifacts/models/"
