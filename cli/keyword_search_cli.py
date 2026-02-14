#!/usr/bin/env python3

import argparse
from lib.keywork_search import search_command
from lib.InvertedIndex import build_command,tf_command,idf_command,tfidf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    build_parser = subparsers.add_parser("build", help="Build the Inverted Index")
    
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    idf_parser = subparsers.add_parser("idf", help="Get the inverse term frequency")
    idf_parser.add_argument("query", type=str, help="Search query")
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Get the TF-IDF")
    tfidf_parser.add_argument("doc_id",type=int, help="Document ID")
    tfidf_parser.add_argument("query", type=str, help="Search query")
    
    tf_parser = subparsers.add_parser("tf", help="Get the term frequency")
    tf_parser.add_argument("doc_id",type=int, help="Document ID")
    tf_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:

        case "search":      
            result=search_command(args.query,5)
            if result:
                for i,r in enumerate(result):
                    print(f'{i}. {r["title"]}')
            else:
                print(f"No Movie Found for {args.query} keyword.")
        case "build": 
            build_command()
        
        case "idf":
            idf_command(args.query)
        case "tf": 
            doc_id=args.doc_id
            query=args.query
            result=tf_command(doc_id=doc_id,term=query)
            print(f'Term frequency is : {result}')
        case "tfidf":
            tfidf_command(args.query,args.doc_id)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
