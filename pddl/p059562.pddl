(define (problem logistics-test)
(:domain logistics)
(:objects
	apn4 apn3 apn6 apn5 - airplane
	cit3 cit4 cit1 cit2 cit6 - city
	apt1 apt5 apt2 apt8 - airport
	tru5 tru3 tru2 tru1 tru4 - truck
	obj55 obj77 obj44 obj11 obj00 obj88 obj99 obj33 obj23 obj66 - package
	pos77 pos21 pos55 pos44 pos66 pos13 pos33 pos23 pos11 pos12 - location
)
(:init
	(at apn4 apt1)
	(at apn3 apt5)
	(at apn6 apt1)
	(at apn5 apt8)
	(at obj55 pos11)
	(at obj77 pos44)
	(at obj44 pos44)
	(at obj11 pos23)
	(at obj00 pos66)
	(at obj88 pos11)
	(at obj99 pos21)
	(at obj33 pos21)
	(at obj23 pos33)
	(at obj66 pos12)
	(in-city apt1 cit1)
	(in-city apt5 cit6)
	(in-city apt2 cit4)
	(in-city apt8 cit2)
	(in-city pos77 cit4)
	(in-city pos21 cit3)
	(in-city pos55 cit2)
	(in-city pos44 cit3)
	(in-city pos66 cit3)
	(in-city pos13 cit6)
	(in-city pos33 cit6)
	(in-city pos23 cit4)
	(in-city pos11 cit4)
	(in-city pos12 cit3)
	(at tru4 pos77)
	(at tru1 pos44)
	(at tru2 pos55)
	(at tru3 pos33)
	(at tru5 pos44)
)
(:goal
	(and
		(at obj11 pos77)
		(at obj33 pos12)
))
)
