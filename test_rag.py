#!/usr/bin/env python3
"""
Test RAG integration
"""

import asyncio
from rag_integration import get_rag_health, initialize_rag, process_rag_query

async def test_rag():
    print('Testing RAG initialization...')
    success = await initialize_rag()
    if success:
        print('RAG initialized successfully!')
        health = await get_rag_health()
        print('RAG Health Status:')
        for key, value in health.items():
            print(f'  {key}: {value}')
        
        # Test a simple query
        print('\nTesting RAG query...')
        result = await process_rag_query('How many facts do we have?')
        print('Query Result:')
        for key, value in result.items():
            if key != 'sources':  # Skip sources for brevity
                print(f'  {key}: {value}')
        print(f'  sources: {len(result.get("sources", []))} source(s)')
    else:
        print('RAG initialization failed')

if __name__ == "__main__":
    asyncio.run(test_rag())