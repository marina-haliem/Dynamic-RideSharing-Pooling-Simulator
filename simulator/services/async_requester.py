"""Asynchronous request module"""
from concurrent import futures
import itertools
import requests

class AsyncRequester(object):
    """Send asynchronous requests to list of urls
    Response time may be limited by rate of system/NW"""
    def __init__(self, n_threads):
        # self.urllist = []
        self.n_threads = n_threads
        self.executor = futures.ThreadPoolExecutor(max_workers=self.n_threads)

    def send_async_requests(self, urllist):
        """Sends asynchronous requests"""
        if len(urllist) == 1:
            return self.get_batch(urllist)
        # Num of batches per thread
        n_batches = int(len(urllist) / self.n_threads) + 1
        # List of URLs per batch
        batch_urllist = [urllist[i * n_batches: (i + 1) * n_batches]
                        for i in range(self.n_threads)]
        # List of HTTP response
        responses = list(self.executor.map(self.get_batch, batch_urllist))
        return list(itertools.chain(*responses))    # Return 1 sequence of responses

    def get_json(cls, url):
        """open URL and return JSON contents"""
        result = requests.get(url).json()
        return result

    def get_batch(self, urllist):
        """Batch processing for get method; takes list of urls as input"""
        return [self.get_json(url) for url in urllist]
