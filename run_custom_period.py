#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
기간 지정 실행 스크립트
사용자가 원하는 기간으로 시점별 실행을 쉽게 할 수 있도록 도와주는 스크립트
"""

import sys
import os
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from example import run_custom_period_execution

def main():
    """기간 지정 실행 메인 함수"""
    print("=== 텍스트 마이닝 기반 기업-토픽 연관도 스코어링 시스템 ===")
    print("기간을 지정하여 시점별 실행을 시작합니다.")
    print()
    
    # 사용자 입력 받기
    try:
        execution_name = input("실행명을 입력하세요 (기본값: custom_execution): ").strip()
        if not execution_name:
            execution_name = "custom_execution"
        
        start_date = input("시작일을 입력하세요 (YYYY-MM-DD 형식, 예: 2021-12-01): ").strip()
        if not start_date:
            print("시작일이 입력되지 않았습니다.")
            return
        
        end_date = input("종료일을 입력하세요 (YYYY-MM-DD 형식, 예: 2024-12-01): ").strip()
        if not end_date:
            print("종료일이 입력되지 않았습니다.")
            return
        
        rebalancing_months_input = input("리밸런싱 주기를 입력하세요 (개월, 기본값: 6): ").strip()
        if rebalancing_months_input:
            try:
                rebalancing_months = int(rebalancing_months_input)
            except ValueError:
                print("잘못된 숫자 형식입니다. 기본값 6을 사용합니다.")
                rebalancing_months = 6
        else:
            rebalancing_months = 6
        
        # 날짜 형식 검증
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            print("날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요.")
            return
        
        # 시작일이 종료일보다 늦은지 확인
        if datetime.strptime(start_date, '%Y-%m-%d') >= datetime.strptime(end_date, '%Y-%m-%d'):
            print("시작일이 종료일보다 늦거나 같습니다. 올바른 날짜를 입력해주세요.")
            return
        
        print()
        print("=== 실행 정보 확인 ===")
        print(f"실행명: {execution_name}")
        print(f"시작일: {start_date}")
        print(f"종료일: {end_date}")
        print(f"리밸런싱 주기: {rebalancing_months}개월")
        print()
        
        # 실행 확인
        confirm = input("위 정보로 실행하시겠습니까? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("실행이 취소되었습니다.")
            return
        
        print()
        print("=== 시점별 실행 시작 ===")
        
        # 실행
        execution_manager = run_custom_period_execution(
            execution_name=execution_name,
            start_date=start_date,
            end_date=end_date,
            rebalancing_months=rebalancing_months
        )
        
        if execution_manager:
            print()
            print("=== 실행 완료 ===")
            status = execution_manager.get_execution_status()
            print(f"실행명: {status['execution_name']}")
            print(f"총 시점 수: {status['total_periods']}")
            print(f"완료된 시점 수: {status['completed_periods']}")
            print(f"결과 폴더: {status['results_path']}")
            print()
            print("결과 파일을 확인해보세요!")
        else:
            print("실행 중 오류가 발생했습니다.")
    
    except KeyboardInterrupt:
        print("\n실행이 중단되었습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()
