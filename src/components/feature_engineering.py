"""
Module : feature_engineering.py
Description : This module contains the FeatureEngineering class which is responsible for creating features from the URL.
"""

import pandas as pd
import re
import math
from collections import Counter
from urllib.parse import urlparse
import dns.resolver
from pyspark.sql.functions import udf, col, length, regexp_replace
from pyspark.sql.types import FloatType, IntegerType, StructType, StructField
from pyspark.sql import DataFrame

from src import logger

sensitive_words = ['secure', 'account', 'webscr', 'login', 'ebayisapi', 'signin']

def subdomain_level(url : str) -> int:
    """
    Calculate the subdomain level of the URL
    
    Args:
        url (str): The URL
    
    Returns:
        int: The subdomain level
    """
    try:
        hostname = urlparse(url).hostname
        if hostname:
            return len(hostname.split('.')) - 2
        return 0
    except Exception as e:
        logger.error(f"Error in subdomain_level: {e}")
        return 0

def path_level(url : str) -> int:
    """
    Calculate the path level of the URL
    
    Args:
        url (str): The URL
    
    Returns:
        int: The path level
    """
    try:
        path = urlparse(url).path
        return len(path.split('/')) - 1
    except Exception as e:
        logger.error(f"Error in path_level: {e}")
        return 0

def count_dashes_hostname(url : str) -> int:
    """
    Count the number of dashes in the hostname of the URL.
    
    Args:
        url (str): The URL
    
    Returns:
        int: The number of dashes in the hostname
    """
    try:
        hostname = urlparse(url).hostname
        if hostname:
            return hostname.count('-')
        return 0
    except Exception as e:
        logger.error(f"Error in count_dashes_hostname: {e}")
        return 0

def no_https(url : str) -> int:
    """
    Check if the URL does not have HTTPS
    
    Args:
        url (str): The URL
    
    Returns:
        int: 1 if the URL does not have HTTPS, 0 otherwise
    """
    try:
        return 0 if urlparse(url).scheme == 'https' else 1
    except Exception as e:
        logger.error(f"Error in no_https: {e}")
        return 0

def random_string(url : str) -> int:
    """
    Check if the URL has a random string of characters
    
    Args:
        url (str): The URL
    
    Returns:
        int: 1 if the URL has a random string of characters, 0 otherwise
    """
    try:
        return 1 if re.search(r'[a-zA-Z]{10,}', url) else 0
    except Exception as e:
        logger.error(f"Error in random_string: {e}")
        return 0

def ip_address(url : str) -> int:
    """
    Check if the URL has an IP address
    
    Args:
        url (str): The URL
    
    Returns:
        int: 1 if the URL has an IP address, 0 otherwise
    """
    try:
        return 1 if re.match(r'^(http|https):\/\/\d+\.\d+\.\d+\.\d+', url) else 0
    except Exception as e:
        logger.error(f"Error in ip_address: {e}")
        return 0

def domain_in_subdomains(url : str) -> int:
    """
    Check if the domain is in the subdomains of the URL
    
    Args:
        url (str): The URL
    
    Returns:
        int: 1 if the domain is in the subdomains, 0 otherwise
    """
    try:
        hostname = urlparse(url).hostname
        if hostname:
            domain_parts = hostname.split('.')
            return 1 if len(domain_parts) > 2 else 0
        return 0
    except Exception as e:
        logger.error(f"Error in domain_in_subdomains: {e}")
        return 0

def domain_in_paths(url : str) -> int:
    """
    Check if the domain is in the paths of the URL
    
    Args:
        url (str): The URL
    
    Returns:
        int: 1 if the domain is in the paths, 0 otherwise
    """
    try:
        path = urlparse(url).path
        hostname = urlparse(url).hostname
        if hostname and path:
            return 1 if hostname in path else 0
        return 0
    except Exception as e:
        logger.error(f"Error in domain_in_paths: {e}")
        return 0

def https_in_hostname(url : str) -> int:
    """
    Check if the URL has HTTPS in the hostname
    
    Args:
        url (str): The URL
    
    Returns:
        int: 1 if the URL has HTTPS in the hostname, 0 otherwise
    """
    try:
        hostname = urlparse(url).hostname
        return 1 if hostname and 'https' in hostname else 0
    except Exception as e:
        logger.error(f"Error in https_in_hostname: {e}")
        return 0

def hostname_length(url : str) -> int:
    """
    Calculate the length of the hostname of the URL
    
    Args:
        url (str): The URL
    
    Returns:
        int: The length of the hostname
    """
    try:
        hostname = urlparse(url).hostname
        return len(hostname) if hostname else 0
    except Exception as e:
        logger.error(f"Error in hostname_length: {e}")
        return 0

def count_sensitive_words(url : str) -> int:
    """
    Count the number of sensitive words in the URL
    
    Args:
        url (str): The URL
    
    Returns:
        int: The number of sensitive words in the URL
    """
    try:
        return sum(word in url.lower() for word in sensitive_words)
    except Exception as e:
        logger.error(f"Error in count_sensitive_words: {e}")
        return 0

def calculate_entropy(url : str) -> float:
    """
    Calculate the entropy of the URL
    
    Args:
        url (str): The URL
    
    Returns:
        float: The entropy of the URL
    """
    try:
        counter = Counter(url)
        total_len = len(url)
        entropy = -sum((count / total_len) * math.log2(count / total_len) for count in counter.values())
        return entropy
    except Exception as e:
        logger.error(f"Error in calculate_entropy: {e}")
        return 0

def ensure_scheme(url : str) -> str:
    """
    Ensure the URL has a scheme
    
    Args:
        url (str): The URL
    
    Returns:
        str: The URL with a scheme
    """
    try:
        if not urlparse(url).scheme:
            return 'http://' + url
        return url
    except Exception as e:
        logger.error(f"Error in ensure_scheme: {e}")
        return url

def get_dns_info(url : str) -> tuple:
    """
    Get DNS information for the URL
    
    Args:
        url (str): The URL
    
    Returns:
        tuple: The DNS information for the URL
    """
    try:
        url = ensure_scheme(url)
        domain = urlparse(url).hostname
        if not domain:
            return (0, 0, 0, 0, 0)

        answers = dns.resolver.resolve(domain, 'A')
        ip_address = answers[0].address if answers else None
        return (len(answers), len(ip_address.split('.')), sum(int(x) for x in ip_address.split('.')), 1 if ip_address else 0, len(domain))
    except Exception as e:
        logger.error(f"Error retrieving DNS data for {url}: {e}")
        return (0, 0, 0, 0, 0)
    
    
class FeatureEngineering:

    def __init__(self, spark):
        self.special_chars = ['@','?','-','=','.','#','%','+','$','!','*',',','//', '&']
        self.schema = StructType([
            StructField("num_dns_answers", IntegerType(), True),
            StructField("num_ip_octets", IntegerType(), True),
            StructField("sum_ip_octets", IntegerType(), True),
            StructField("has_ip_address", IntegerType(), True),
            StructField("domain_length", IntegerType(), True)
        ])
        self.udfs = self._register_udfs()
        self.spark = spark

    def _register_udfs(self):
        return {
            'subdomain_level': udf(subdomain_level, IntegerType()),
            'path_level': udf(path_level, IntegerType()),
            'count_dashes_hostname': udf(count_dashes_hostname, IntegerType()),
            'no_https': udf(no_https, IntegerType()),
            'random_string': udf(random_string, IntegerType()),
            'ip_address': udf(ip_address, IntegerType()),
            'domain_in_subdomains': udf(domain_in_subdomains, IntegerType()),
            'domain_in_paths': udf(domain_in_paths, IntegerType()),
            'https_in_hostname': udf(https_in_hostname, IntegerType()),
            'hostname_length': udf(hostname_length, IntegerType()),
            'count_sensitive_words': udf(count_sensitive_words, IntegerType()),
            'calculate_entropy': udf(calculate_entropy, FloatType()),
        }

    def create_features(self, df_spark : DataFrame) -> DataFrame:
        """
        Create features from the URL
        
        Args:
            df_spark (DataFrame): The DataFrame containing the URL
        
        Returns:
            DataFrame: The DataFrame with the features
        """
        try:
            df_spark = df_spark.withColumn('url_length', length(col('url')))
            df_spark = df_spark.withColumn('num_dots', length(regexp_replace(col('url'), '[^.]', '')) - 1)
            df_spark = df_spark.withColumn('num_hyphens', length(regexp_replace(col('url'), '[^-]', '')) - 1)
            df_spark = df_spark.withColumn('num_digits', length(regexp_replace(col('url'), '[^\d]', '')))
            df_spark = df_spark.withColumn('num_alpha', length(regexp_replace(col('url'), '[^a-zA-Z]', '')))
            df_spark = df_spark.withColumn('subdomain_level', self.udfs['subdomain_level']('url'))
            df_spark = df_spark.withColumn('path_level', self.udfs['path_level']('url'))
            df_spark = df_spark.withColumn('count_dashes_hostname', self.udfs['count_dashes_hostname']('url'))
            df_spark = df_spark.withColumn('no_https', self.udfs['no_https']('url'))
            df_spark = df_spark.withColumn('random_string', self.udfs['random_string']('url'))
            df_spark = df_spark.withColumn('ip_address', self.udfs['ip_address']('url'))
            df_spark = df_spark.withColumn('domain_in_subdomains', self.udfs['domain_in_subdomains']('url'))
            df_spark = df_spark.withColumn('domain_in_paths', self.udfs['domain_in_paths']('url'))
            df_spark = df_spark.withColumn('https_in_hostname', self.udfs['https_in_hostname']('url'))
            df_spark = df_spark.withColumn('hostname_length', self.udfs['hostname_length']('url'))
            df_spark = df_spark.withColumn('count_sensitive_words', self.udfs['count_sensitive_words'](col('url')))
            df_spark = df_spark.withColumn('url_entropy', self.udfs['calculate_entropy'](col('url')))
            
            for char in self.special_chars:
                df_spark = df_spark.withColumn(f'num_{char}', length(regexp_replace(col('url'), f'[^{char}]', '')))
                
            df_spark = df_spark.withColumnRenamed('num_.', 'num_dot')
            df_spark = df_spark.withColumnRenamed('num_#', 'num_hash')
            df_spark = df_spark.withColumnRenamed('num_%', 'num_percent')
            df_spark = df_spark.withColumnRenamed('num_+', 'num_plus')
            df_spark = df_spark.withColumnRenamed('num_$', 'num_dollar')
            df_spark = df_spark.withColumnRenamed('num_!', 'num_exclamation')
            df_spark = df_spark.withColumnRenamed('num_*', 'num_asterisk')
            df_spark = df_spark.withColumnRenamed('num_,', 'num_comma')
            df_spark = df_spark.withColumnRenamed('num_&', 'num_query_symbol')

            return df_spark
        except Exception as e:
            logger.error(f"Error in create_features: {e}")
            raise

    def add_dns_features(self, df_spark: DataFrame) -> DataFrame:
        """
        Add DNS features to the DataFrame
        
        Args:
            df_spark (DataFrame): The DataFrame containing the URL
        
        Returns:
            DataFrame: The DataFrame with the DNS features
        """
        try:
            dns_data = []
            urls = [row['url'] for row in df_spark.collect()]
            for url in urls:
                dns_info = get_dns_info(url)
                dns_data.append({'url': url, 'num_dns_answers': dns_info[0], 'num_ip_octets': dns_info[1], 'sum_ip_octets': dns_info[2], 'has_ip_address': dns_info[3], 'domain_length': dns_info[4]})
            
            dns_info_df = pd.DataFrame(dns_data)
            dns_info_spark_df = self.spark.createDataFrame(dns_info_df)
            
            df_spark = df_spark.join(dns_info_spark_df, on="url", how="left")
            return df_spark
        except Exception as e:
            logger.error(f"Error in add_dns_features: {e}")
            raise