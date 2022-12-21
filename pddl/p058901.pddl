(define (problem logistics-test)
(:domain logistics)
(:objects
	apn2 apn4 apn3 apn8 - airplane
	cit3 cit5 cit6 cit1 cit4 - city
	apt3 apt7 apt1 apt6 - airport
	tru2 tru1 tru4 tru3 tru5 - truck
	obj12 obj66 obj00 obj11 obj21 obj33 obj23 obj88 obj55 obj77 - package
	pos11 pos33 pos12 pos13 pos23 pos21 pos66 pos22 pos44 pos55 - location
)
(:init
	(at apn2 apt1)
	(at apn4 apt3)
	(at apn3 apt7)
	(at apn8 apt1)
	(at obj12 pos33)
	(at obj66 pos11)
	(at obj00 pos66)
	(at obj11 pos55)
	(at obj21 pos55)
	(at obj33 pos33)
	(at obj23 pos23)
	(at obj88 pos13)
	(at obj55 pos55)
	(at obj77 pos55)
	(in-city apt3 cit5)
	(in-city apt7 cit6)
	(in-city apt1 cit1)
	(in-city apt6 cit4)
	(in-city pos11 cit6)
	(in-city pos33 cit1)
	(in-city pos12 cit3)
	(in-city pos13 cit1)
	(in-city pos23 cit6)
	(in-city pos21 cit5)
	(in-city pos66 cit1)
	(in-city pos22 cit5)
	(in-city pos44 cit1)
	(in-city pos55 cit4)
	(at tru5 pos23)
	(at tru3 pos66)
	(at tru4 pos12)
	(at tru1 pos22)
	(at tru2 pos55)
)
(:goal
	(and
		(at obj33 pos33)
		(at obj11 pos44)
))
)
