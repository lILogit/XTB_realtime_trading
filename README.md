This program is the backbone for univesal online trading on the XTB platform using the API. The loop is waiting for the tick price change and based on the given strategy places orders (ENTER, SL, TP, EXIT).

1. XTB credentials
  modify /utils/secrets.py 

        class credentials():
            """
            https://co.xtb.com/#/demo-accounts
            """
            XTB_DEMO_ID = "***"
            XTB_PASS_KEY = "***"
            XTB_ACCOUNT_TYPE = "demo" or "real"
  

2. Trading SYMBOL (from xtb list)
         eg. SYMBOL = "EURUSD"
   
3. Init bot - Client login, global variables, daily pivots, create thread etc.
       
        class Worker(QThread):
             def __init__(self, parent=None):

4. Predefined daily entry signals / pivot prices (enter, SL, TP, exit)
   
        class signal:
            def __init__(self, type, sl, enter, exit):
                self.type = type
                self.sl = sl
                self.enter = enter
                self.exit = exit
   
6. Tick price change indicator, actual price = object['ask']

       if (last_price != prev_price): logger.info("Price change: {}", object['ask'])

7. Order trigger - cross entry signal/price 

        # cross trigger status
                for obj in signals:
                    trigger_status = 0
                    if (last_price <= obj.enter and prev_price > obj.enter) or (
                            last_price >= obj.enter and prev_price < obj.enter):
                        trigger_status = 1
                        break

8. Send order

          self.send_order(obj.type, obj.sl, obj.enter, obj.exit, last_price)
   
10. Bot info/error logs - bot_{time}.log

          logger.add("bot_{time}.log") #,level="INFO")
