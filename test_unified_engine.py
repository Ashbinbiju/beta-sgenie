#!/usr/bin/env python3
"""
Test script for UnifiedAnalysisEngine
Tests the new unified analysis system with sample stocks
"""

import sys
sys.path.insert(0, '/workspaces/beta-sgenie')

from streamlit_app import UnifiedAnalysisEngine
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_unified_engine(symbol, trading_style='swing'):
    """Test the UnifiedAnalysisEngine with a stock"""
    print(f"\n{'='*80}")
    print(f"Testing {symbol} - {trading_style.upper()} Trading")
    print(f"{'='*80}\n")
    
    try:
        # Create engine instance
        engine = UnifiedAnalysisEngine(
            symbol=symbol,
            trading_style=trading_style,
            timeframe='15min' if trading_style == 'swing' else '5min',
            account_size=30000
        )
        
        # Run analysis
        result = engine.analyze()
        
        # Display results
        print(f"📊 ANALYSIS RESULTS FOR {result['symbol']}")
        print(f"   Trading Style: {result['trading_style']}")
        print(f"   Timeframe: {result['timeframe']}")
        print(f"\n🎯 RECOMMENDATION")
        print(f"   Overall Score: {result['overall_score']:.1f}/100")
        print(f"   Signal: {result['signal']}")
        print(f"   Confidence: {result['confidence']}%")
        
        print(f"\n📈 COMPONENT SCORES")
        for component, score in result['scores'].items():
            print(f"   {component.replace('_', ' ').title()}: {score:.1f}/100")
        
        print(f"\n💪 STRENGTHS ({len(result['strengths'])})")
        for strength in result['strengths'][:5]:
            print(f"   • {strength}")
        
        print(f"\n🎯 OPPORTUNITIES ({len(result['opportunities'])})")
        for opp in result['opportunities'][:5]:
            print(f"   • {opp}")
        
        print(f"\n⚠️  WARNINGS ({len(result['warnings'])})")
        for warning in result['warnings'][:5]:
            print(f"   • {warning}")
        
        if result.get('threats'):
            print(f"\n🚨 THREATS ({len(result['threats'])})")
            for threat in result['threats'][:5]:
                print(f"   • {threat}")
        
        print(f"\n📊 MARKET CONTEXT")
        mc = result['market_context']
        print(f"   Health: {mc['health']:.0f}/100")
        print(f"   Signal: {mc['signal']}")
        
        print(f"\n📰 NEWS SUMMARY")
        ns = result['news_summary']
        print(f"   Available: {ns['available']}")
        print(f"   Count: {ns['count']}")
        print(f"   Sentiment Score: {ns['sentiment_score']:.1f}/100")
        
        print(f"\n🎯 TECHNICAL LEVELS")
        tl = result['technical_levels']
        print(f"   Current Price: ₹{tl['current_price']:.2f}")
        print(f"   Pivot: ₹{tl['pivot']:.2f}")
        print(f"   Support: ₹{tl['support'][0]:.2f}, ₹{tl['support'][1]:.2f}, ₹{tl['support'][2]:.2f}")
        print(f"   Resistance: ₹{tl['resistance'][0]:.2f}, ₹{tl['resistance'][1]:.2f}, ₹{tl['resistance'][2]:.2f}")
        
        print(f"\n✅ TEST PASSED for {symbol}")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED for {symbol}")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test stocks
    test_stocks = [
        ("SBIN-EQ", "swing"),
        ("RELIANCE-EQ", "swing"),
        ("HDFCBANK-EQ", "intraday"),
    ]
    
    print("\n" + "="*80)
    print("UNIFIED ANALYSIS ENGINE TEST SUITE")
    print("="*80)
    
    results = []
    for symbol, style in test_stocks:
        success = test_unified_engine(symbol, style)
        results.append((symbol, style, success))
        print("\n" + "-"*80 + "\n")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for symbol, style, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {symbol} ({style})")
    
    passed = sum(1 for _, _, s in results if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED")
        sys.exit(1)
