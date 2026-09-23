//+------------------------------------------------------------------+
//| ExportResearchData.mq5                                           |
//| Exports what the Python research needs from YOUR broker terminal: |
//|   research/symbols.csv   every symbol with its trading specs      |
//|   research/sessions.csv  trading sessions of the chosen symbols   |
//|   research/bars_<SYM>_M1.csv   M1 history with spread (points)    |
//|   research/calendar_high.csv   high-impact economic events        |
//| Files go to <Data Folder>/MQL5/Files/research/                    |
//| (File > Open Data Folder in MT5).                                 |
//|                                                                  |
//| Before running: Tools > Options > Charts > "Max bars in chart" =  |
//| Unlimited, otherwise M1 history is cut short.                    |
//+------------------------------------------------------------------+
#property copyright "tradelab"
#property version   "1.00"
#property script_show_inputs

input string   InpSymbols         = "EURUSD,GBPJPY,XAUUSD,NAS100,USOIL,BTCUSD,USDJPY"; // symbols to export (exact broker names)
input datetime InpFrom            = D'2018.01.01';
input datetime InpTo              = D'2030.01.01';
input bool     InpExportBars      = true;
input bool     InpExportCalendar  = true;
input string   InpCalendarCcys    = "USD,EUR,GBP,JPY";

const string DIR = "research\\";

//+------------------------------------------------------------------+
string Iso(datetime t)
  {
   MqlDateTime d;
   TimeToStruct(t, d);
   return StringFormat("%04d-%02d-%02d %02d:%02d:%02d", d.year, d.mon, d.day, d.hour, d.min, d.sec);
  }

//+------------------------------------------------------------------+
int SplitList(string csv, string &out[])
  {
   int n = StringSplit(csv, ',', out);
   for(int i = 0; i < n; i++)
     {
      StringTrimLeft(out[i]);
      StringTrimRight(out[i]);
     }
   return n;
  }

//+------------------------------------------------------------------+
//| All symbols on the server, so you can find the exact names.      |
//+------------------------------------------------------------------+
void ExportSymbols()
  {
   int h = FileOpen(DIR + "symbols.csv", FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE) { Print("symbols.csv: open failed ", GetLastError()); return; }
   FileWrite(h, "symbol", "description", "path", "digits", "point", "contract_size",
             "volume_min", "volume_step", "volume_max", "currency_base", "currency_profit",
             "currency_margin", "calc_mode", "swap_mode", "swap_long", "swap_short",
             "swap_3day", "tick_size", "tick_value", "spread_now", "stops_level",
             "margin_1lot_buy");
   int total = SymbolsTotal(false);
   for(int i = 0; i < total; i++)
     {
      string s = SymbolName(i, false);
      double margin = 0.0;
      double ask = SymbolInfoDouble(s, SYMBOL_ASK);
      if(ask > 0)
         if(!OrderCalcMargin(ORDER_TYPE_BUY, s, 1.0, ask, margin))
            margin = 0.0;
      FileWrite(h, s,
                SymbolInfoString(s, SYMBOL_DESCRIPTION),
                SymbolInfoString(s, SYMBOL_PATH),
                (int)SymbolInfoInteger(s, SYMBOL_DIGITS),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_POINT), 10),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_TRADE_CONTRACT_SIZE), 4),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_VOLUME_MIN), 4),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_VOLUME_STEP), 4),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_VOLUME_MAX), 2),
                SymbolInfoString(s, SYMBOL_CURRENCY_BASE),
                SymbolInfoString(s, SYMBOL_CURRENCY_PROFIT),
                SymbolInfoString(s, SYMBOL_CURRENCY_MARGIN),
                EnumToString((ENUM_SYMBOL_CALC_MODE)SymbolInfoInteger(s, SYMBOL_TRADE_CALC_MODE)),
                EnumToString((ENUM_SYMBOL_SWAP_MODE)SymbolInfoInteger(s, SYMBOL_SWAP_MODE)),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_SWAP_LONG), 4),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_SWAP_SHORT), 4),
                EnumToString((ENUM_DAY_OF_WEEK)SymbolInfoInteger(s, SYMBOL_SWAP_ROLLOVER3DAYS)),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_TRADE_TICK_SIZE), 10),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_TRADE_TICK_VALUE), 6),
                (int)SymbolInfoInteger(s, SYMBOL_SPREAD),
                (int)SymbolInfoInteger(s, SYMBOL_TRADE_STOPS_LEVEL),
                DoubleToString(margin, 2));
     }
   FileClose(h);
   PrintFormat("symbols.csv: %d symbols", total);
  }

//+------------------------------------------------------------------+
void ExportSessions(string &syms[])
  {
   int h = FileOpen(DIR + "sessions.csv", FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE) { Print("sessions.csv: open failed ", GetLastError()); return; }
   FileWrite(h, "symbol", "weekday", "from", "to");
   for(int k = 0; k < ArraySize(syms); k++)
      for(int d = 0; d < 7; d++)
         for(uint idx = 0; idx < 10; idx++)
           {
            datetime from, to;
            if(!SymbolInfoSessionTrade(syms[k], (ENUM_DAY_OF_WEEK)d, idx, from, to))
               break;
            FileWrite(h, syms[k], EnumToString((ENUM_DAY_OF_WEEK)d),
                      TimeToString(from, TIME_MINUTES), TimeToString(to, TIME_MINUTES));
           }
   FileClose(h);
  }

//+------------------------------------------------------------------+
//| M1 bars, exported month by month to keep memory use small.       |
//+------------------------------------------------------------------+
void ExportBars(string sym)
  {
   if(!SymbolSelect(sym, true))
     {
      PrintFormat("%s: symbol not found on this server, skipped", sym);
      return;
     }
   int h = FileOpen(DIR + "bars_" + sym + "_M1.csv", FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE) { Print(sym, ": open failed ", GetLastError()); return; }
   FileWrite(h, "time", "open", "high", "low", "close", "tick_volume", "spread");
   int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   long written = 0;
   datetime first = 0;
   datetime chunk = InpFrom;
   datetime stop = (InpTo < TimeCurrent()) ? InpTo : TimeCurrent();
   while(chunk < stop && !IsStopped())
     {
      MqlDateTime d;
      TimeToStruct(chunk, d);
      d.mon += 1;
      if(d.mon > 12) { d.mon = 1; d.year += 1; }
      d.day = 1; d.hour = 0; d.min = 0; d.sec = 0;
      datetime next = StructToTime(d);
      MqlRates r[];
      int n = CopyRates(sym, PERIOD_M1, chunk, next - 1, r);
      for(int i = 0; i < n; i++)
        {
         if(first == 0) first = r[i].time;
         FileWrite(h, Iso(r[i].time),
                   DoubleToString(r[i].open, digits), DoubleToString(r[i].high, digits),
                   DoubleToString(r[i].low, digits), DoubleToString(r[i].close, digits),
                   (long)r[i].tick_volume, r[i].spread);
        }
      if(n > 0) written += n;
      chunk = next;
     }
   FileClose(h);
   PrintFormat("%s: %I64d M1 bars from %s", sym, written, (first > 0) ? Iso(first) : "n/a");
  }

//+------------------------------------------------------------------+
//| High-impact events. Times are trade-server time, as the EA sees. |
//+------------------------------------------------------------------+
void ExportCalendar()
  {
   int h = FileOpen(DIR + "calendar_high.csv", FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE) { Print("calendar: open failed ", GetLastError()); return; }
   FileWrite(h, "time", "currency", "impact", "event");
   string ccys[];
   int nc = SplitList(InpCalendarCcys, ccys);
   int total = 0;
   for(int c = 0; c < nc; c++)
     {
      MqlCalendarValue values[];
      datetime cal_to = TimeTradeServer() + 30 * 86400;
      if(InpTo < cal_to) cal_to = InpTo;
      if(CalendarValueHistory(values, InpFrom, cal_to, NULL, ccys[c]) <= 0)
        {
         PrintFormat("calendar %s: no data (error %d)", ccys[c], GetLastError());
         continue;
        }
      for(int i = 0; i < ArraySize(values); i++)
        {
         MqlCalendarEvent ev;
         if(!CalendarEventById(values[i].event_id, ev)) continue;
         if(ev.importance != CALENDAR_IMPORTANCE_HIGH) continue;
         string name = ev.name;
         StringReplace(name, ",", " ");
         FileWrite(h, Iso(values[i].time), ccys[c], "high", name);
         total++;
        }
     }
   FileClose(h);
   PrintFormat("calendar_high.csv: %d high-impact events", total);
  }

//+------------------------------------------------------------------+
void OnStart()
  {
   string syms[];
   SplitList(InpSymbols, syms);
   ExportSymbols();
   ExportSessions(syms);
   if(InpExportCalendar) ExportCalendar();
   if(InpExportBars)
      for(int k = 0; k < ArraySize(syms) && !IsStopped(); k++)
         ExportBars(syms[k]);
   Print("Done. Files are in MQL5/Files/research/ (File > Open Data Folder).");
  }
//+------------------------------------------------------------------+
